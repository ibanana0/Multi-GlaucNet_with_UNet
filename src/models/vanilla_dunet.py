"""
Vanilla DUNet: A deformable network for retinal vessel & optic disc segmentation.
Faithful implementation of Qiangguo Jin et al. (Knowledge-Based Systems, 2019)
optimized with torchvision.ops.deform_conv2d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DeformConvBlock(nn.Module):
    """
    Deformable Convolutional Layer (DCNv2) dengan learned 2D offsets dan modulation mask.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Convolution pembelajar offset spasial 2D (2 * k * k)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        # Convolution pembelajar mask modulasi bobot (k * k)
        self.mask_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        # Bobot konvolusi deformable
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Inisialisasi
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask
        )


class DoubleConv(nn.Module):
    """Standard (Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DoubleDeformConv(nn.Module):
    """Deformable (DeformConv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = DeformConvBlock(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = DeformConvBlock(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class StandardDown(nn.Module):
    """Downscaling dengan MaxPool2d(2) diikuti DoubleConv standar"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class DeformDown(nn.Module):
    """Downscaling dengan MaxPool2d(2) diikuti DoubleDeformConv"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleDeformConv(in_ch, out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class StandardUp(nn.Module):
    """Upscaling (bilinear/convTranspose) diikuti skip-connection dan DoubleConv standar"""
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DeformUp(nn.Module):
    """Upscaling diikuti skip-connection dan DoubleDeformConv"""
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleDeformConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class VanillaDUNet(nn.Module):
    """
    Arsitektur DUNet murni persis sesuai DUNet-retinal-vessel-detection (Jin et al., 2019):
    - Level 0: inc (standard double conv) -> 64 ch
    - Level 1: down1 (deformable down) -> 128 ch
    - Level 2: down2 (deformable down) -> 256 ch
    - Level 3: down3 (standard down) -> 512 ch
    - Level 4: down4 (standard down) -> 512 ch
    - Level 3: up1 (standard up) -> 256 ch
    - Level 2: up2 (standard up) -> 128 ch
    - Level 1: up3 (deformable up) -> 64 ch
    - Level 0: up4 (deformable up) -> 64 ch
    - Signature Skip: torch.cat([inp, x], dim=1)
    - Output: 1x1 Conv -> Sigmoid
    """
    def __init__(self, in_channels: int = 3, out_classes: int = 1, filter_factor: int = 1):
        super().__init__()
        c = 64 // filter_factor

        self.inc = DoubleConv(in_channels, c)
        self.down1 = DeformDown(c, c * 2)
        self.down2 = DeformDown(c * 2, c * 4)
        self.down3 = StandardDown(c * 4, c * 8)
        self.down4 = StandardDown(c * 8, c * 8)

        self.up1 = StandardUp(c * 16, c * 4)
        self.up2 = StandardUp(c * 8, c * 2)
        self.up3 = DeformUp(c * 4, c)
        self.up4 = DeformUp(c * 2, c)

        # Concatenation dengan input asli (c + in_channels)
        self.outc = nn.Conv2d(c + in_channels, out_classes, kernel_size=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(inp)      # 512x512, c
        x2 = self.down1(x1)     # 256x256, 2c
        x3 = self.down2(x2)     # 128x128, 4c
        x4 = self.down3(x3)     # 64x64, 8c
        x5 = self.down4(x4)     # 32x32, 8c

        x = self.up1(x5, x4)    # 64x64, 4c
        x = self.up2(x, x3)     # 128x128, 2c
        x = self.up3(x, x2)     # 256x256, c
        x = self.up4(x, x1)     # 512x512, c

        # Skip connection dari citra input asli khas DUNet
        x = torch.cat([inp, x], dim=1)  # 512x512, c + in_channels
        logits = self.outc(x)
        return torch.sigmoid(logits)
