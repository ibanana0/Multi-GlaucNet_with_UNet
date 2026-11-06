"""
Visualization and Qualitative Evaluation Module for MultiGlaucNet:
Plots original fundus image, predicted OD and Vessel masks,
masked inputs (I_OD and I_vessel), and detection probabilities.
"""

import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import torch


def save_segmentation_prediction_plot(
    fundus_img: Union[np.ndarray, torch.Tensor],
    pred_od: Union[np.ndarray, torch.Tensor],
    pred_vessel: Union[np.ndarray, torch.Tensor],
    gt_od: Optional[Union[np.ndarray, torch.Tensor]] = None,
    gt_vessel: Optional[Union[np.ndarray, torch.Tensor]] = None,
    save_path: str = "results/visualizations/segmentation_sample.png"
):
    """
    Menyimpan plot perbandingan citra asli, masker prediksi, dan ground-truth.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Konversi torch tensor ke numpy jika perlu
    def to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            if x.ndim == 3 and x.shape[0] in [1, 3]:
                x = np.transpose(x, (1, 2, 0))
            if x.ndim == 3 and x.shape[2] == 1:
                x = x.squeeze(-1)
        return x

    img = to_np(fundus_img)
    pod = to_np(pred_od)
    pvess = to_np(pred_vessel)

    cols = 5 if (gt_od is not None and gt_vessel is not None) else 3
    fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4))

    axes[0].imshow(img)
    axes[0].set_title("Citra Fundus Asli")
    axes[0].axis("off")

    axes[1].imshow(pod, cmap="gray")
    axes[1].set_title("Prediksi Optic Disc (DUNet)")
    axes[1].axis("off")

    axes[2].imshow(pvess, cmap="gray")
    axes[2].set_title("Prediksi Vessel (DUNet)")
    axes[2].axis("off")

    if cols == 5:
        axes[3].imshow(to_np(gt_od), cmap="gray")
        axes[3].set_title("Ground Truth OD")
        axes[3].axis("off")

        axes[4].imshow(to_np(gt_vessel), cmap="gray")
        axes[4].set_title("Ground Truth Vessel")
        axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Visualisasi segmentasi disimpan ke: {save_path}")


def save_three_branch_overview_plot(
    I_orig: Union[np.ndarray, torch.Tensor],
    I_od: Union[np.ndarray, torch.Tensor],
    I_vessel: Union[np.ndarray, torch.Tensor],
    pred_label: str,
    confidence: float,
    true_label: Optional[str] = None,
    save_path: str = "results/visualizations/three_branch_sample.png"
):
    """
    Menyimpan plot Three-Branch input yang masuk ke ResNet50:
    Branch 1: Global Retina, Branch 2: Local OD, Branch 3: Local Vessel.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    def to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            if x.ndim == 3 and x.shape[0] == 3:
                x = np.transpose(x, (1, 2, 0))
        return np.clip(x, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(to_np(I_orig))
    axes[0].set_title("Branch 1: Global Retinal Context (I_orig)")
    axes[0].axis("off")

    axes[1].imshow(to_np(I_od))
    axes[1].set_title("Branch 2: Local OD Context (I_OD)")
    axes[1].axis("off")

    axes[2].imshow(to_np(I_vessel))
    axes[2].set_title("Branch 3: Local Vessel Context (I_vessel)")
    axes[2].axis("off")

    title_text = f"Deteksi: {pred_label} (Confidence: {confidence*100:.1f}%)"
    if true_label is not None:
        title_text += f" | Ground Truth: {true_label}"
    plt.suptitle(title_text, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Visualisasi Three-Branch disimpan ke: {save_path}")
