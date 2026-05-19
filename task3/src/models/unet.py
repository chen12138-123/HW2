from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2):
        if channels % g == 0:
            return g
    return 1


def _norm_layer(kind: Literal["bn", "gn"], channels: int) -> nn.Module:
    if kind == "gn":
        return nn.GroupNorm(_gn_groups(channels), channels)
    return nn.BatchNorm2d(channels)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: Literal["bn", "gn"] = "bn") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm_layer(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm_layer(norm, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: Literal["bn", "gn"] = "bn") -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: Literal["bn", "gn"] = "bn") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, norm=norm)

    @staticmethod
    def _pad_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        dh = ref.shape[-2] - x.shape[-2]
        dw = ref.shape[-1] - x.shape[-1]
        if dh == 0 and dw == 0:
            return x
        pad_left = dw // 2
        pad_right = dw - pad_left
        pad_top = dh // 2
        pad_bottom = dh - pad_top
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._pad_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 64, norm: Literal["bn", "gn"] = "bn") -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels, norm=norm)
        self.down1 = Down(base_channels, base_channels * 2, norm=norm)
        self.down2 = Down(base_channels * 2, base_channels * 4, norm=norm)
        self.down3 = Down(base_channels * 4, base_channels * 8, norm=norm)
        self.down4 = Down(base_channels * 8, base_channels * 16, norm=norm)

        self.up1 = Up(base_channels * 16, base_channels * 8, norm=norm)
        self.up2 = Up(base_channels * 8, base_channels * 4, norm=norm)
        self.up3 = Up(base_channels * 4, base_channels * 2, norm=norm)
        self.up4 = Up(base_channels * 2, base_channels, norm=norm)
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 3
    num_classes: int = 8
    base_channels: int = 64
    norm: Literal["bn", "gn"] = "bn"

    def build(self) -> UNet:
        return UNet(self.in_channels, self.num_classes, self.base_channels, norm=self.norm)
