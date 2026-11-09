"""
Hybrid DUNet: Novelty Architecture fusing Deformable Convolutional Network (DCNv2)
into a ResNet Backbone with a 5-Stage PixelShuffle Decoder and Channel Attention Mechanism (CAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from .vanilla_dunet import DeformConvBlock


class ChannelAttention(nn.Module):
    """
    Channel Attention Mechanism (Squeeze-and-Excitation block)
    secara dinamis mengalibrasi pembobotan channel fitur untuk menonjolkan
    sinyal tepi Optic Disc & vaskular serta menekan derau background.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        reduced_ch = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_ch, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) dari CBAM (Woo et al., 2018):
    Mengekstrak informasi spasial inter-channel melalui pooling rata-rata (avg)
    dan pooling maksimum (max), diikuti konvolusi 7x7 untuk menghasilkan
    spatial attention map berbobot [0, 1] yang berfokus pada kurvatur vaskular & kontur OD.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018):
    Menggabungkan Channel Attention Mechanism (CAM) secara berurutan
    dengan Spatial Attention Module (SAM) untuk menyeleksi fitur anatomis optimal.
    """
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class DeformableBottleneck(nn.Module):
    """
    ResNet Bottleneck Block di mana konvolusi 3x3 digantikan oleh Deformable Convolution (DCNv2).
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        mid_ch = out_ch // 4

        # 1x1 conv
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        # 3x3 Deformable Conv
        self.conv2 = DeformConvBlock(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        # 1x1 conv
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class PixelShuffleUpBlock(nn.Module):
    """
    Decoder block yang memanfaatkan Sub-Pixel Convolution (PixelShuffle)
    diikuti Concatenation dengan skip connection, CBAM (Channel + Spatial Attention), dan Conv-BN-ReLU.
    """
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int = 0):
        super().__init__()
        # PixelShuffle(2) memperkecil channel 4x dan memperbesar HxW 2x
        # Jadi conv1 memperbesar channel ke out_ch * 4
        self.pre_shuffle_conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.bn_shuffle = nn.BatchNorm2d(out_ch)
        self.relu_shuffle = nn.ReLU(inplace=True)

        total_ch = out_ch + skip_ch
        # CBAM Block: Channel Attention diikuti Spatial Attention
        self.attention = CBAMBlock(total_ch)

        self.post_conv = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.relu_shuffle(self.bn_shuffle(self.pixel_shuffle(self.pre_shuffle_conv(x))))
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
        x = self.attention(x)
        return self.post_conv(x)


class HybridDUNet(nn.Module):
    """
    Arsitektur Hibrida Teroptimasi (Novelty):
    - Encoder: Deformable ResNet Backbone (DCNv2 Bottlenecks)
    - Decoder: 5-Stage PixelShuffle + CBAM (Channel & Spatial Attention)
    - High-Resolution Input Skip Connection pada stage 5 (up5) untuk menjaga keutuhan kapiler mikro 1 piksel.
    - Memulihkan resolusi spasial penuh 512x512 secara eksak tanpa artefak checkerboard.
    """
    def __init__(self, in_channels: int = 3, out_classes: int = 1, backbone: str = "resnet50"):
        super().__init__()

        # Konfigurasi kedalaman ResNet (ResNet-50: [3, 4, 6, 3], ResNet-101: [3, 4, 23, 3])
        layers = [3, 4, 6, 3] if backbone == "resnet50" else [3, 4, 23, 3]

        # Stem awal (512x512 -> 256x256 -> 128x128)
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder Stages (DCN Bottlenecks)
        self.stage1 = self._make_stage(64, 256, layers[0], stride=1)    # 128x128
        self.stage2 = self._make_stage(256, 512, layers[1], stride=2)   # 64x64
        self.stage3 = self._make_stage(512, 1024, layers[2], stride=2)  # 32x32
        self.stage4 = self._make_stage(1024, 2048, layers[3], stride=2) # 16x16

        # Decoder Stages (5-Stage PixelShuffle + CBAM)
        # Stage 1 Up: 16x16 (2048) -> 32x32 (512) + skip stage3 (1024)
        self.up1 = PixelShuffleUpBlock(in_ch=2048, out_ch=512, skip_ch=1024)
        # Stage 2 Up: 32x32 (512) -> 64x64 (256) + skip stage2 (512)
        self.up2 = PixelShuffleUpBlock(in_ch=512, out_ch=256, skip_ch=512)
        # Stage 3 Up: 64x64 (256) -> 128x128 (128) + skip stage1 (256)
        self.up3 = PixelShuffleUpBlock(in_ch=256, out_ch=128, skip_ch=256)
        # Stage 4 Up: 128x128 (128) -> 256x256 (64) + skip pre-pool stem (64)
        self.up4 = PixelShuffleUpBlock(in_ch=128, out_ch=64, skip_ch=64)
        # Stage 5 Up: 256x256 (64) -> 512x512 (32) + High-Resolution Input Skip Connection (in_channels)
        self.up5 = PixelShuffleUpBlock(in_ch=64, out_ch=32, skip_ch=in_channels)

        # Output Final Conv 1x1 -> Sigmoid
        self.final_conv = nn.Conv2d(32, out_classes, kernel_size=1)

    def _make_stage(self, in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        layers = [DeformableBottleneck(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(DeformableBottleneck(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simpan tensor citra input asli untuk koneksi pintas resolusi tinggi (High-Res Skip)
        x_input = x

        # Encoder
        x_prepool = self.stem_conv(x)  # 256x256, 64
        x0 = self.stem_pool(x_prepool) # 128x128, 64
        x1 = self.stage1(x0)           # 128x128, 256
        x2 = self.stage2(x1)           # 64x64, 512
        x3 = self.stage3(x2)           # 32x32, 1024
        x4 = self.stage4(x3)           # 16x16, 2048

        # Decoder (5 Stages Rekonstruksi Penuh dengan CBAM)
        d1 = self.up1(x4, skip=x3)        # 32x32, 512
        d2 = self.up2(d1, skip=x2)        # 64x64, 256
        d3 = self.up3(d2, skip=x1)        # 128x128, 128
        d4 = self.up4(d3, skip=x_prepool) # 256x256, 64
        d5 = self.up5(d4, skip=x_input)   # 512x512, 32 (High-Resolution Input Skip Connection)

        logits = self.final_conv(d5)
        return torch.sigmoid(logits)
