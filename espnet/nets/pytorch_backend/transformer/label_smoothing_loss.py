#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module."""

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
        self.reduction_type = criterion.reduction # Save original reduction for 'mean' or 'sum'

    def forward(self, x, target, reduction: str = "mean"):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :param str reduction: "mean", "sum", or "none"
        :return: scalar float value or tensor if reduction is 'none'
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x_view = x.view(-1, self.size)
        target_view = target.view(-1)

        with torch.no_grad():
            true_dist = x_view.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target_view == self.padding_idx  # (B*seqlen,)
            total = len(target_view) - ignore.sum().item()
            target_masked = target_view.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target_masked.unsqueeze(1), self.confidence)

        # self.criterion is nn.KLDivLoss(reduction="none") by default
        kl = self.criterion(torch.log_softmax(x_view, dim=1), true_dist)
        # kl shape is (batch*seqlen, C) if criterion.reduction is 'none' and C > 1
        # or (batch*seqlen,) if C=1 (not the case here for KLDiv)
        # Sum over classes, result shape (batch*seqlen,)
        kl = kl.sum(dim=1)

        kl_masked = kl.masked_fill(ignore, 0)

        if reduction == "none":
            # Reshape to (batch_size, seq_len) and sum over seq_len to get per-sample loss
            # This assumes that individual sequence losses are summed up.
            # If normalize_length is true, it should be handled by the caller.
            # Or, if we want per-token average loss for each sample:
            # kl_reshaped = kl_masked.view(batch_size, -1)
            # seq_lengths = (target != self.padding_idx).sum(dim=1)
            # per_sample_loss = kl_reshaped.sum(dim=1) / seq_lengths.clamp(min=1)
            # return per_sample_loss
            # For now, returning sum of losses per sample, as this is common for weighting.
            return kl_masked.view(batch_size, -1).sum(dim=1)

        # Original logic for 'mean' or 'sum'
        # Note: self.criterion.reduction was 'none'. If we want to match original 'mean'/'sum'
        # behavior of a typical criterion, we'd do it here.
        # However, the original code sums and then divides by denom.
        summed_loss = kl_masked.sum()

        if self.normalize_length:
            denom = total if total > 0 else 1 # Avoid division by zero
        else:
            denom = batch_size if batch_size > 0 else 1 # Avoid division by zero

        if reduction == "mean":
            return summed_loss / denom
        elif reduction == "sum":
            return summed_loss
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}")
