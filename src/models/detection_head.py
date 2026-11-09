"""
Three-Branch ResNet50 Detection Head for Glaucoma Classification
according to MultiGlaucNet architecture:
- Branch 1: Global Retinal Context (I_orig)
- Branch 2: Local Optic Disc Context (I_OD = I_orig * M_OD)
- Branch 3: Local Retinal Vessel Context (I_vessel = I_orig * M_vessel)
- Fusion: Elementwise Average Pooling -> FC (Normal vs Glaucoma)
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights


class ThreeBranchResNet50(nn.Module):
    """
    Tiga cabang ResNet50 independen untuk memproses:
    1. Konteks global retina (I_orig)
    2. Konteks lokal Optic Disc (I_OD = I_orig * M_OD)
    3. Konteks lokal jaringan pembuluh darah (I_vessel = I_orig * M_vessel)
    
    Args:
        apply_imagenet_norm: Jika True, menerapkan normalisasi ImageNet (z-score) pada input.
                             Default False untuk menjaga kompatibilitas 100% dengan checkpoint
                             eksisting yang dilatih pada skala [0, 1].
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, fusion_method: str = "average", apply_imagenet_norm: bool = False):
        super().__init__()
        self.fusion_method = fusion_method.lower()
        self.apply_imagenet_norm = apply_imagenet_norm

        if self.apply_imagenet_norm:
            self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        weights = ResNet50_Weights.DEFAULT if pretrained else None

        # 3 Cabang ResNet50
        self.branch_orig = resnet50(weights=weights)
        self.branch_od = resnet50(weights=weights)
        self.branch_vessel = resnet50(weights=weights)

        # Lepaskan linear classifier asli dari ResNet50 (gunakan feature extractor hingga avgpool)
        self.feature_dim = self.branch_orig.fc.in_features  # 2048
        self.branch_orig.fc = nn.Identity()
        self.branch_od.fc = nn.Identity()
        self.branch_vessel.fc = nn.Identity()

        # Classifier setelah fusi
        if self.fusion_method == "concat":
            classifier_in_dim = self.feature_dim * 3
        else:  # "average" (default sesuai arsitektur.jpg dan XML)
            classifier_in_dim = self.feature_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(classifier_in_dim, num_classes)
        )

    def extract_features(self, I_orig: torch.Tensor, I_od: torch.Tensor, I_vessel: torch.Tensor):
        x_orig = self.normalize(I_orig) if self.apply_imagenet_norm else I_orig
        x_od = self.normalize(I_od) if self.apply_imagenet_norm else I_od
        x_vessel = self.normalize(I_vessel) if self.apply_imagenet_norm else I_vessel

        f_orig = self.branch_orig(x_orig)      # [B, 2048]
        f_od = self.branch_od(x_od)            # [B, 2048]
        f_vessel = self.branch_vessel(x_vessel) # [B, 2048]
        return f_orig, f_od, f_vessel

    def forward(self, I_orig: torch.Tensor, I_od: torch.Tensor, I_vessel: torch.Tensor) -> torch.Tensor:
        f_orig, f_od, f_vessel = self.extract_features(I_orig, I_od, I_vessel)

        if self.fusion_method == "concat":
            f_fused = torch.cat([f_orig, f_od, f_vessel], dim=1)  # [B, 6144]
        else:
            # Sesuai diagram arsitektur.jpg: lingkaran Average (f_orig + f_od + f_vessel) / 3
            f_fused = (f_orig + f_od + f_vessel) / 3.0           # [B, 2048]

        logits = self.classifier(f_fused)  # [B, num_classes]
        return logits
