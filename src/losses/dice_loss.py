"""
Dice Loss and BCE-Dice Loss for Retinal Segmentation (Optic Disc & Blood Vessels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss untuk segmentasi biner:
    DiceLoss = 1 - (2 * sum(p_i * g_i) + smooth) / (sum(p_i) + sum(g_i) + smooth)
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Asumsikan pred sudah melewati fungsi sigmoid (skala 0 s/d 1)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Kombinasi Binary Cross-Entropy dan Dice Loss untuk stabilitas gradien:
    Loss = BCE + DiceLoss
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp pred untuk mencegah log(0) pada BCELoss
        pred_clamped = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        loss_bce = self.bce(pred_clamped, target)
        loss_dice = self.dice(pred, target)
        return (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)


class TverskyLoss(nn.Module):
    """
    Tversky Loss (Salehi et al., 2017) untuk menangani ketidakseimbangan kelas ekstrem:
    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    Loss = 1 - Tversky
    Dengan beta > alpha (misal alpha=0.3, beta=0.7), penalti False Negative lebih tinggi
    sehingga model lebih agresif mendeteksi pembuluh kapiler mikro yang tipis.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1.0 - target_flat)).sum()
        fn = ((1.0 - pred_flat) * target_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (Abraham & Khan, 2019):
    Loss = (1 - Tversky)^gamma
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 4.0 / 3.0, smooth: float = 1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(pred, target)
        return torch.pow(tversky_loss, self.gamma)
