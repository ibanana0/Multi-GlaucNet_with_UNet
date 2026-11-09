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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Visualisasi segmentasi disimpan ke: {save_path} (DPI 300)")


def save_contour_overlay_plot(
    fundus_img: Union[np.ndarray, torch.Tensor],
    pred_od: Union[np.ndarray, torch.Tensor],
    pred_vessel: Union[np.ndarray, torch.Tensor],
    save_path: str = "results/visualizations/contour_overlay_sample.png"
):
    """
    Menyimpan plot overlay kontur anatomis beresolusi publikasi (DPI 300):
    - Kontur Optic Disc digambar dengan warna kuning cerah (Yellow #FFE600).
    - Kontur Pembuluh Darah digambar dengan warna cyan cerah (Cyan #00F0FF).
    """
    import cv2
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    def to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            if x.ndim == 3 and x.shape[0] in [1, 3]:
                x = np.transpose(x, (1, 2, 0))
            if x.ndim == 3 and x.shape[2] == 1:
                x = x.squeeze(-1)
        return x

    img = to_np(fundus_img).copy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    pod = (to_np(pred_od) > 0.5).astype(np.uint8)
    pvess = (to_np(pred_vessel) > 0.5).astype(np.uint8)

    # Buat overlay
    overlay = img.copy()

    # Ekstraksi dan gambar kontur Optic Disc (Kuning)
    contours_od, _ = cv2.findContours(pod, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_od, -1, (255, 230, 0), thickness=2)

    # Ekstraksi dan gambar kontur Pembuluh Darah (Cyan)
    contours_vessel, _ = cv2.findContours(pvess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_vessel, -1, (0, 240, 255), thickness=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Citra Fundus Asli (Enhanced CLAHE)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Overlay Kontur DUNet (Kuning: OD | Cyan: Vessel)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Visualisasi kontur overlay disimpan ke: {save_path} (DPI 300)")


def plot_multimodel_roc_curve(
    roc_dict: dict,
    save_path: str = "results/visualizations/multimodel_roc_curve.png",
    title: str = "Perbandingan Kurva ROC Deteksi Glaukoma (SMDG-19 Test Set)"
):
    """
    Menyimpan plot perbandingan kurva ROC multi-skenario (DPI 300):
    Args:
        roc_dict: Dictionary dengan format {
            "Label Model/Skenario": {"fpr": [...], "tpr": [...], "auc": float, "color": str}
        }
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))

    # Garis acuan chance diagonal
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Random Guess (AUC = 0.5000)")

    for model_name, data in roc_dict.items():
        fpr = data["fpr"]
        tpr = data["tpr"]
        auc_val = data.get("auc", 0.0)
        color = data.get("color", None)
        line_style = data.get("linestyle", "-")
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.4f})", linewidth=2.0, color=color, linestyle=line_style)

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Plot Kurva ROC Multi-Model disimpan ke: {save_path} (DPI 300)")


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
    Menyimpan plot Three-Branch input yang masuk ke ResNet50 (DPI 300):
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Visualisasi Three-Branch disimpan ke: {save_path} (DPI 300)")
