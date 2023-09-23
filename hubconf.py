"""torch.hub configuration."""

dependencies = ["torch",]

from typing import Callable               # pylint: disable=wrong-import-position

import torch                              # pylint: disable=wrong-import-position
from torch import Tensor                  # pylint: disable=wrong-import-position

from vqwav2vec.model import VQWav2VecUnit # pylint: disable=wrong-import-position
from vqwav2vec.pad import pad_for_vqw2v   # pylint: disable=wrong-import-position

URLS = {
    "vqw2v_unit": "https://github.com/tarepan/Infer-vq-wav2vec/releases/download/v1.0.0/vqw2v_unit_v1.pt",
}
# [Origin]
# "vqw2v_unit" is derived from s3prl/s3prl 'vq_wav2vec_gumbel' checkpoint, under Apache License 2.0 (Copyright 2022 Andy T. Liu and Shu-wen Yang, https://github.com/s3prl/s3prl/blob/main/LICENSE).
# Weight transfer code is in this repository (`/vqwav2vec/transfer/weight.py`).


def vqw2v_unit(progress: bool = True) -> VQWav2VecUnit:
    """
    `vq-wav2vec` speech-to-unit inference model.

    Args:
        progress - Whether to show model checkpoint load progress
    """

    state_dict = torch.hub.load_state_dict_from_url(url=URLS["vqw2v_unit"], map_location="cpu", progress=progress)
    model = VQWav2VecUnit()
    model.load_state_dict(state_dict)
    model.eval()

    return model


def vqw2v_pad() -> Callable[[Tensor], Tensor]:
    """Padding utility function for `vqw2v_unit`."""
    return pad_for_vqw2v
