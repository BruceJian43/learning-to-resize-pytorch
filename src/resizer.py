import torch
import torch.nn as nn

from typing import Tuple, Union
from functools import partial

def conv1x1(in_chs: int, out_chs: int = 16) -> nn.Conv2d:
    return nn.Conv2d(
        in_chs,
        out_chs,
        kernel_size=1,
        stride=1,
        padding=0
    )

def conv3x3(in_chs: int, out_chs: int = 16) -> nn.Conv2d:
    return nn.Conv2d(
        in_chs,
        out_chs,
        kernel_size=3,
        stride=1,
        padding=1
    )

def conv7x7(in_chs: int, out_chs: int = 16) -> nn.Conv2d:
    return nn.Conv2d(
        in_chs,
        out_chs,
        kernel_size=7,
        stride=1,
        padding=3
    )

class ResBlock(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int = 16,
    ) -> None:
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_chs, out_chs),
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU(0.2),
            conv3x3(out_chs, out_chs),
            nn.BatchNorm2d(out_chs)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.layers(x)
        out += identity
        return out

class Resizer(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_size: Union[int, Tuple[int, int]],
        n_filters: int = 16,
        n_res_blocks: int = 1,
        mode: str = 'bilinear'
    ) -> None:
        super(Resizer, self).__init__()
        self.interpolate_layer = partial(
            nn.functional.interpolate,
            size=out_size,
            mode=mode,
            align_corners=(True if mode in ('linear', 'bilinear', 'bicubic', 'trilinear') else None)
        )
        self.conv_layers = nn.Sequential(
            conv7x7(in_chs, n_filters),
            nn.LeakyReLU(0.2),
            conv1x1(n_filters, n_filters),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(n_filters)
        )
        self.residual_layers = nn.Sequential()
        for i in range(n_res_blocks):
            self.residual_layers.add_module(
                f'res{i}',
                ResBlock(n_filters, n_filters)
            )
        self.residual_layers.add_module('conv3x3', conv3x3(n_filters, n_filters))
        self.residual_layers.add_module('bn', nn.BatchNorm2d(n_filters))
        self.final_conv = conv7x7(n_filters, in_chs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.interpolate_layer(x)
        conv_out = self.conv_layers(x)
        conv_out = self.interpolate_layer(conv_out)
        conv_out_identity = conv_out
        res_out = self.residual_layers(conv_out)
        res_out += conv_out_identity
        out = self.final_conv(res_out)
        out += identity
        return out
