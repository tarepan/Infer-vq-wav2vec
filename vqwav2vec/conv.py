"""Convolutional modules."""

from torch import nn, Tensor


class DownConv(nn.Module):
    """[Conv-GN-ReLU]x8-LogCompression."""

    def __init__(self):
        """Init."""
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        def gen_layer(c_in: int, c_out: int, kernel: int, stride: int) -> nn.Module:
            """Generate 'Conv1d-GN-ReLU' layer."""
            return nn.Sequential(*[
                nn.Conv1d(c_in, c_out, kernel, stride=stride, bias=False),
                nn.GroupNorm(1, c_out, affine=True),
                nn.ReLU(),
            ])

        # 8 layers, k10/k8/k4/k4/k4/k1/k1/k1
        conv_layers = [
        #    c_i  c_o   k  s
            (  1, 512, 10, 5),
            (512, 512,  8, 4),
            (512, 512,  4, 2),
            (512, 512,  4, 2),
            (512, 512,  4, 2),
            (512, 512,  1, 1),
            (512, 512,  1, 1),
            (512, 512,  1, 1),
        ]
        self.conv_layers = nn.Sequential(*[gen_layer(c_in, c_out, kernel, stride) for c_in, c_out, kernel, stride in conv_layers])

    def forward(self, series: Tensor) -> Tensor:
        """Downsampled Conv + Log compression :: (B, T) -> (B, Feat=1, T) -> (B, Feat, Frame) -> (B, Feat, Frame)."""
        series = self.conv_layers(series.unsqueeze(1))
        return (series.abs() + 1).log()
