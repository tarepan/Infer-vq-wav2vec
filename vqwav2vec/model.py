"""vq-wav2vec unit inference."""

from torch import nn, Tensor

from .conv import DownConv
from .vq import GumbelGroupVQ


class VQWav2VecUnit(nn.Module):
    """vq-wav2vec unit extraction.
    
    In total, this model works as 'wave16k -> [Conv_k465_s160] -> [GroupedVQ_g2_subcode320] -> unit_100Hz_feat512'
    """

    def __init__(self):
        """Init."""
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        self.down_conv = DownConv()
        self.quant = GumbelGroupVQ()

    def forward(self, wave: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward a batch.

        Args:
            wave         :: (B, T)               - Audio waveform, sampling rate should be 16kHz
        Returns:
            z_series     :: (B, Frame=frm100hz, Feat=512) - Continuous latent       series
            q_series     :: (B, Frame=frm100hz, Feat=512) - Discrete   latent       series
            q_idx_series :: (B, Frame=frm100hz)           - Discrete   latent index series
        """

        z_series = self.down_conv(wave)
        q_series, q_idx_series = self.quant(z_series)

        return z_series.transpose(1, 2), q_series.transpose(1, 2), q_idx_series
