"""
Focal Loss implementation for Glaucoma Classification
supporting the 3 Ablation Study scenarios:
- Scenario 1 (Baseline): gamma = 0.0, alpha = 1.0 (Standard Cross-Entropy)
- Scenario 2 (Standard Literature): gamma = 2.0, alpha = empirical ratio (Lin et al.)
- Scenario 3 (Extreme Paper): gamma = 10.0, alpha = 3.0 (Multi-GlaucNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class FocalLoss(nn.Module):
    """
    Focal Loss untuk klasifikasi multikelas / biner:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[Union[float, List[float], torch.Tensor]] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (int, float)):
            # Jika skalar float (misal alpha=3.0 untuk glaukoma, 1.0 untuk normal)
            self.alpha = torch.tensor([1.0, float(alpha)], dtype=torch.float32)
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.float()
        else:
            raise TypeError(f"Format alpha tidak didukung: {type(alpha)}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Logits luaran jaringan [B, C] (sebelum softmax)
            targets: Label ground-truth [B] dengan nilai kelas 0 s/d C-1
        """
        # Hitung log softmax dan probabilitas softmax
        log_p = F.log_softmax(logits, dim=-1)           # [B, C]
        p = torch.exp(log_p)                            # [B, C]

        # Ambil probabilitas kelas target
        targets = targets.view(-1, 1)                   # [B, 1]
        log_pt = log_p.gather(1, targets).squeeze(-1)   # [B]
        pt = p.gather(1, targets).squeeze(-1)           # [B]

        # Hitung modulasi focal (1 - p_t)^gamma
        focal_term = (1.0 - pt) ** self.gamma

        loss = -focal_term * log_pt

        # Terapkan bobot kelas alpha jika ada
        if self.alpha is not None:
            alpha_device = self.alpha.to(logits.device)
            alpha_t = alpha_device.gather(0, targets.squeeze(-1))
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
