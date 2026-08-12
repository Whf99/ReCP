"""Compact 2D U-Net used only to expose the public backbone contract.

This inspection model is not the parameter-matched experimental network reported
in the manuscript and must not be used to reproduce its parameter/FLOP results.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    """Return evidence logits and a feature map through the public model API."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.bridge = ConvBlock(base * 2, base * 4)
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bridge = self.bridge(self.pool(e2))
        d2 = F.interpolate(bridge, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.dec1(torch.cat((d1, e1), dim=1))
        evidence = F.softplus(self.head(features))
        return {"evidence": evidence, "features": features}
