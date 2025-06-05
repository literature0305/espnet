#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module."""

from typing import Optional # For utt_weights

import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
    ):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target, utt_weights: Optional[torch.Tensor] = None):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :param Optional[torch.Tensor] utt_weights:
            utterance-level weights (batch)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        # Original x shape: (batch_size, seq_len, self.size)
        # Original target shape: (batch_size, seq_len)

        x_flat = x.view(-1, self.size)
        target_flat = target.view(-1)

        with torch.no_grad():
            true_dist = x_flat.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target_flat == self.padding_idx  # shape (batch_size * seq_len)
            total = len(target_flat) - ignore.sum().item()
            target_flat_masked = target_flat.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target_flat_masked.unsqueeze(1), self.confidence)

        kl = self.criterion(torch.log_softmax(x_flat, dim=1), true_dist)
        # kl shape: (batch_size * seq_len, self.size)

        # Calculate element-wise losses and mask padded elements
        elementwise_loss = kl.masked_fill(ignore.unsqueeze(1), 0)
        # elementwise_loss shape: (batch_size * seq_len, self.size)

        # Sum losses over the vocabulary dimension
        loss_summed_over_vocab = elementwise_loss.sum(dim=1)
        # loss_summed_over_vocab shape: (batch_size * seq_len)

        # Reshape to separate batch and sequence
        # target.size(1) is the original sequence length
        seq_len = target.size(1)
        loss_per_token = loss_summed_over_vocab.view(batch_size, seq_len)
        # loss_per_token shape: (batch_size, seq_len)

        # Sum losses for each utterance
        loss_per_utterance = loss_per_token.sum(dim=1)
        # loss_per_utterance shape: (batch_size)

        # Apply utterance weights if provided
        if utt_weights is not None:
            if utt_weights.size(0) != batch_size:
                raise ValueError(
                    f"utt_weights has batch size {utt_weights.size(0)}, "
                    f"but loss_per_utterance has batch size {batch_size}"
                )
            loss_per_utterance = loss_per_utterance * utt_weights.to(
                loss_per_utterance.device
            ).type_as(loss_per_utterance)

        total_loss = loss_per_utterance.sum()

        denom = total if self.normalize_length else batch_size
        return total_loss / denom
