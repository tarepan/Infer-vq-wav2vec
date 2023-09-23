"""Padding."""

from torch import Tensor
import torch.nn.functional as F


def pad_for_vqw2v(wave: Tensor) -> Tensor:
    """Same padding with frame centering.

    Args:
        wave :: (..., T=t)     - Waveform
    Returns:
             :: (..., T=l+t+r) - Padded waveform
    """

    kernel, hop = 465, 160
    padding_total = kernel - hop
    padding_l = padding_total // 2
    padding_r = padding_total - padding_l
    padded_wave = F.pad(wave, (padding_l, padding_r), mode="constant")

    return padded_wave
