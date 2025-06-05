import math
from typing import Collection, Dict, List, Tuple, Union

import numpy as np
import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.nets_utils import pad_list
import scipy.signal # Added for HuBERTCollateFn, ensure it's here if not already
import soundfile # Added for HuBERTCollateFn


class CommonCollateFn:
    """Functor class of common_collate_fn()"""

    @typechecked
    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
    ):
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return common_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


class HuBERTCollateFn(CommonCollateFn):
    """Functor class of common_collate_fn()"""

    @typechecked
    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        label_downsampling: int = 1,
        pad: bool = False,
        rand_crop: bool = True,
        crop_audio: bool = True,
        not_sequence: Collection[str] = (),
        window_size: float = 25,
        window_shift: float = 20,
        sample_rate: float = 16,
        noise_scp: str = "data/noise/wav.scp",
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "-5_20",
        dynamic_mixing_gain_db: float = 5.0,
        dynamic_mixing_prob=0.1,
        mix_speech: bool = False,
        reverb_speech: bool = False,
        rir_scp: str = "data/rirs/wav.scp",
        rir_apply_prob: float = 0.3,
        train: bool = True,
    ):
        super().__init__(
            float_pad_value=float_pad_value,
            int_pad_value=int_pad_value,
            not_sequence=not_sequence,
        )
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.label_downsampling = label_downsampling
        self.pad = pad
        self.rand_crop = rand_crop
        self.crop_audio = crop_audio
        self.not_sequence = set(not_sequence)
        self.window_size = window_size
        self.window_shift = window_shift
        self.sample_rate = sample_rate
        self.train = train
        self.mix_speech = mix_speech
        self.reverb_speech = reverb_speech
        self.dynamic_mixing_prob = dynamic_mixing_prob
        self.noise_apply_prob = noise_apply_prob
        self.dynamic_mixing_gain_db = dynamic_mixing_gain_db
        self.rir_apply_prob = rir_apply_prob

        # Load noise data for WavLM-style
        if train and mix_speech and noise_scp is not None:
            self.noises = {}
            self.noise_paths = []
            with open(noise_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        noise_path = sps[0]
                    else:
                        noise_path = sps[1]
                    self.noise_paths.append(noise_path)
            sps = noise_db_range.split("_")
            if len(sps) == 1:
                self.noise_db_low = self.noise_db_high = float(sps[0])
            elif len(sps) == 2:
                self.noise_db_low, self.noise_db_high = float(sps[0]), float(sps[1])
            else:
                raise ValueError(
                    "Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]"
                )
        else:
            self.noises = None

        # Load RIRs for reverberation
        if train and reverb_speech and rir_scp is not None:
            self.rirs = {}
            self.rir_paths = []
            with open(rir_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        rir_path = sps[0]
                    else:
                        rir_path = sps[1]
                    self.rir_paths.append(rir_path)
        else:
            self.rirs = None

    def _read_rir_audio_(self):
        """
        Read RIR audio from a list of paths.
        We cache the audio in memory to reduce I/O.
        """
        rir_path = np.random.choice(self.rir_paths)
        rir = None
        if rir_path is not None:
            if rir_path in self.rirs:
                rir = self.rirs[rir_path]
            else:
                with soundfile.SoundFile(rir_path) as f:
                    rir = f.read(dtype=np.float32, always_2d=False)
                    if rir.ndim == 2:
                        rir = np.mean(rir, axis=1)
                self.rirs[rir_path] = rir
        return rir

    def _read_noise_audio_(self):
        """
        Read noise audio from a list of paths.
        We cache the audio in memory to reduce I/O.
        """
        noise_path = np.random.choice(self.noise_paths)
        noise = None
        if noise_path is not None:
            if noise_path in self.noises:
                noise = self.noises[noise_path]
            else:
                with soundfile.SoundFile(noise_path) as f:
                    noise = f.read(dtype=np.float32, always_2d=False)
                self.noises[noise_path] = noise
        return noise

    def _get_aligned_reverb_signal(self, speech):
        """
        Simulate reverberant audio with a random RIR.

        It is re-aligned to the original signal for
        compatability with HuBERT-style training.

        See https://aclanthology.org/2024.emnlp-main.570/.
        """

        rir = self._read_rir_audio_()
        # speech.shape: [mics=1, samples]
        # rir.shape: [mics=1, samples2]

        speech = speech.reshape(1, -1)
        rir = rir.reshape(1, -1)

        power = (speech[_detect_non_silence(speech)] ** 2).mean() # Use local _detect_non_silence
        dt = np.argmax(rir, axis=1).min()
        speech2 = scipy.signal.convolve(speech, rir, mode="full")[
            :, dt : dt + speech.shape[1]
        ]

        # Reverse mean power to the original power
        power2 = (speech2[_detect_non_silence(speech2)] ** 2).mean() # Use local _detect_non_silence
        speech2 = np.sqrt(power / max(power2, 1e-10)) * speech2

        return speech2.flatten()

    def _add_noise_wavlm(self, data, speech, speech_id):
        """
        WavLM-style augmentation. We randomly choose one of two methods:
            - Denoising -> sample an acoustic noise
            - Separation -> sample another utterance from the batch

        See https://arxiv.org/abs/2110.13900 for details
        """
        power = (speech[_detect_non_silence(speech)] ** 2).mean() # Use local _detect_non_silence
        if self.dynamic_mixing_prob >= np.random.random() or len(data) == 1:
            noise = self._read_noise_audio_().squeeze()
            noise_db = np.random.uniform(self.noise_db_low, self.noise_db_high)
        else:
            noise = random.choice(data)
            while noise[0] == speech_id:
                noise = random.choice(data)
            noise = noise[1]["speech"]
            speech_length = speech.shape[0]
            noise_db = np.random.uniform(
                -self.dynamic_mixing_gain_db, self.dynamic_mixing_gain_db
            )

        length = min(np.random.randint(1, len(speech) // 2 + 1), len(noise))
        speech_start = np.random.randint(0, len(speech) - length + 1)
        noise_start = np.random.randint(0, len(noise) - length + 1)

        noise_power = (noise**2).mean()
        scale = (
            10 ** (-noise_db / 20) * np.sqrt(power) / np.sqrt(max(noise_power, 1e-10))
        )
        noise = noise * scale
        speech[speech_start : speech_start + length] += noise[
            noise_start : noise_start + length
        ]

        return speech

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value}, "
            f"label_downsampling={self.label_downsampling}, "
            f"pad_value={self.pad}, rand_crop={self.rand_crop}) "
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        assert "speech" in data[0][1]
        if self.pad:
            num_frames = max([sample["speech"].shape[0] for uid, sample in data])
        else:
            num_frames = min([sample["speech"].shape[0] for uid, sample in data])

        new_data = []
        if self.train or self.label_downsampling > 1:
            for uid, sample in data:
                waveform = sample["speech"]
                label = sample["text"] if "text" in sample else None

                assert waveform.ndim == 1
                length = waveform.size

                # WavLM Noise
                if (
                    self.train
                    and self.mix_speech
                    and self.noise_apply_prob >= np.random.random()
                ):
                    waveform = self._add_noise_wavlm(data, waveform, uid)

                # Reverberation Augmentation
                if (
                    self.train
                    and self.reverb_speech
                    and self.rir_apply_prob >= np.random.random()
                ):
                    waveform = self._get_aligned_reverb_signal(waveform)

                # The MFCC feature is 10ms per frame, while the transformer output
                # is 20ms per frame. Downsample the KMeans label
                # if it's generated by MFCC features.

                if self.label_downsampling > 1 and label is not None:
                    label = label[:: self.label_downsampling]
                if self.train and self.crop_audio:
                    waveform, label, length = _crop_audio_label(
                        waveform,
                        label,
                        length,
                        num_frames,
                        self.rand_crop,
                        self.window_size,
                        self.window_shift,
                        self.sample_rate,
                    )
                if label is not None:
                    new_data.append((uid, dict(speech=waveform, text=label)))
                else:
                    new_data.append((uid, dict(speech=waveform)))
        else:
            new_data = data

        # Call the common_collate_fn, which will now include utt2weight processing
        # if we modify common_collate_fn directly.
        # However, the original plan was to modify CommonCollateFn.__call__
        # which calls common_collate_fn.
        # For this class, it seems it reimplements some logic and then calls
        # common_collate_fn. So, the utt2weight logic should be in common_collate_fn.
        return common_collate_fn(
            new_data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


def _crop_audio_label(
    waveform: torch.Tensor,
    label: torch.Tensor,
    length: torch.Tensor,
    num_frames: int,
    rand_crop: bool,
    window_size: int = 25,
    window_shift: int = 20,
    sample_rate: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate the audio and label at the same time.

    Args:
        waveform (Tensor): The waveform Tensor with dimensions `(time)`.
        label (Tensor): The label Tensor with dimensions `(seq)`.
        length (Tensor): The length Tensor with dimension `(1,)`.
        num_frames (int): The final length of the waveform.
        rand_crop (bool): if ``rand_crop`` is True, the starting index of the
            waveform and label is random if the length is longer than the minimum
            length in the mini-batch.
        window_size (int): reception field of conv feature extractor (in ms).
            In default, calculated by [400 (samples) / 16 (sample_rate)].
        window_shift (int): the stride of conv feature extractor (in ms).
            In default, calculated by [320 (samples) / 16 (sample_rate)].
        sample_rate (int): number of samples in audio signal per millisecond.

    Returns:
        (Tuple(Tensor, Tensor, Tensor)): Returns the Tensors for the waveform,
            label, and the waveform length.

    """

    frame_offset = 0
    if waveform.size > num_frames and rand_crop:
        diff = waveform.size - num_frames
        frame_offset = torch.randint(diff, size=(1,)) # This should be np.random.randint if waveform is numpy
        if isinstance(waveform, np.ndarray): # Ensure compatibility if waveform is numpy
             frame_offset = np.random.randint(diff)
    elif waveform.size < num_frames:
        num_frames = waveform.size
    label_offset = max(
        math.floor(
            (frame_offset - window_size * sample_rate) / (window_shift * sample_rate)
        )
        + 1,
        0,
    )
    num_label = (
        math.floor(
            (num_frames - window_size * sample_rate) / (window_shift * sample_rate)
        )
        + 1
    )
    waveform = waveform[frame_offset : frame_offset + num_frames]
    if label is not None:
        label = label[label_offset : label_offset + num_label]
    length = num_frames

    return waveform, label, length


# Helper function for HuBERTCollateFn, moved from preprocessor.py to avoid circular import
def _detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)

    framed_w = _framing( # Use local _framing
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    power = (framed_w**2).mean(axis=-1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=bool)
    detect_frames = power / mean_power > threshold
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )

# Helper function for HuBERTCollateFn, moved from preprocessor.py
def _framing(
    x,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1] and x.ndim > 0 : # added x.ndim > 0 to handle 0-dim arrays
        #raise ValueError("frame_length is greater than input length")
        # For HuBERT, it might be just a single frame, so pad it
        if padded:
             x = np.pad(x, [(0,0)]*(x.ndim-1)+[(0, frame_length - x.shape[-1])], mode="constant", constant_values=0)
        else:
            raise ValueError("frame_length is greater than input length and not padded")


    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result


@typechecked
def common_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    uttids = [u for u, _ in data]
    data_dicts = [d for _, d in data] # Renamed to avoid conflict

    assert all(set(data_dicts[0]) == set(d) for d in data_dicts), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data_dicts[0]
    ), f"*_lengths is reserved: {list(data_dicts[0])}"

    output = {}
    for key in data_dicts[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data_dicts[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data_dicts]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data_dicts], dtype=torch.long)
            output[key + "_lengths"] = lens

    # Processing utt2weight
    if data_dicts and 'utt2weight' in data_dicts[0]:
        # Ensure all examples have 'utt2weight' if the first one does, and they are not empty
        # Filter out examples where 'utt2weight' might be missing or empty, though this shouldn't happen with current setup
        utt2weight_values = [d['utt2weight'][0] for d in data_dicts if 'utt2weight' in d and d['utt2weight'].size > 0]
        if utt2weight_values: # Proceed only if the list is not empty
            output['utt2weight'] = torch.tensor(utt2weight_values, dtype=torch.float32)

    output = (uttids, output)
    return output

[end of espnet2/train/collate_fn.py]
