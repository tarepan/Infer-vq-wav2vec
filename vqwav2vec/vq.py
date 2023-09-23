"""Vector quantization. Forked from `Fairseq` under MIT license (https://github.com/facebookresearch/fairseq/blob/main/LICENSE)."""

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class GumbelGroupVQ(nn.Module):
    """Grouped vector quantization with gumbel softmax."""

    def __init__(self):
        """Init."""
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        # [Dev Note]
        # This class handle grouped quantization, so basic type is `(Batch, Time, Group, Subcode, Feat)`.
        # During transformation, Group and Subcode are frequently combined as 'Code'.
        # At output, groups are concatenated so feature dimension becomes Group*Feat=g*feat_g.

        # Params
        self.feat_i    = 512 # Feature dimension size of input series
        self.group     =   2 # The number of groups (the number of sub-codebook; `G`)
        self.code_g    = 320 # The number of codes (representative vectors) per group
        feat_o         = 512 # Feature dimension size of total representative vector
        self.curr_temp = 2.0 # Gumbel-softmax Temperature

        # Feature dimension size of each group's representative vector
        self.code = self.group * self.code_g
        self.feat_g = feat_o // self.group

        # Projection :: (B, T, Feat=feat_i) -> (B, T, Feat=G*V) - FC-ReLU-FC
        self.projection = nn.Sequential(*[
            nn.Linear(self.feat_i, self.feat_i * 2),
            nn.ReLU(),
            nn.Linear(self.feat_i * 2, self.code),
        ])

        # Full Codebook :: (1, Code=gv, Feat=feat_g) - Single codebook which contains all sub-codebooks
        self.codebook = nn.Parameter(torch.empty([1, self.code, self.feat_g], dtype=torch.float))


    def forward(self, series: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize input feature sequences.

        Args:
            series        :: (B=b, Feat=feat_i, T=t) - Input feature series
        Returns:
            q_series      :: (B=b, Feat=feat_o, T=t) - Quantized vector series
            maxidx_series :: (B=b, T=t, Group=g)     - Group argmax indice series (At inference, it is q_series' index)
        """

        # Prepare
        bsz, _, tsz = series.shape

        # Projection :: (B=b, Feat, T=t) -> (B=b, T=t, Feat) -> (BT=bt, Feat) -> (BT=bt, Code=g*v) -> (BTGroup=btg, Subcode=v)
        series = series.transpose(1, 2).reshape(-1, self.feat_i)
        series = self.projection(series)
        series = series.reshape(-1, self.code_g)

        # Sampling

        ## Onehot-nize per group :: (BTGroup=btg, Subcode=v) -> (BTGroup=btg, Subcode=v) -> (BT=bt, Code=gv) - For Argmax sampling
        if not self.training:
            with torch.no_grad():
                series = series.new_zeros(*series.shape).scatter_(-1, series.max(-1)[1].reshape(-1, 1), 1.0).reshape(-1, self.code) # type: ignore
        else:
            series = F.gumbel_softmax(series.float(), tau=self.curr_temp, hard=True).type_as(series).reshape(-1, self.code)

        ## Index-nize :: (BT=bt, Code=gv) -> (BTGroup=btg, Subcode=v) -> (BTGroup=btg,) -> (B=b, T=t, Group=g) - Get index series, in whicj an index is acquired per group by argmax
        maxidx_series = series.reshape(-1, self.code_g).argmax(dim=-1).reshape(bsz, tsz, self.group).detach()

        ## Vector-nize :: (BT, Code) & (1, Code, Feat) -> (B, T, Feat) - one-hot series to total representative vector series
        ### Query :: (BT=bt, Code=gv, 1) * (BT=1, Code=gv, Feat=feat_g) -> (BT=bt, Code=gv, Feat=feat_g) - Query hot-vector, others remain zeros
        series = series.unsqueeze(-1) * self.codebook
        ### Squash & Concat - Squash zero raws in each groups, then concat groups
        #### :: (BT=bt, Code=gv, Feat=feat_g) -> (BT=bt, Group=g, Subcode=v, Feat=feat_g) -> (BT=bt, Group=g, Feat=feat_g) -> (B=b, T=t, Feat=g*feat_g) 
        q_series = series.reshape(-1, self.group, self.code_g, self.feat_g).sum(-2).reshape(bsz, tsz, -1)

        return q_series.transpose(1, 2), maxidx_series
