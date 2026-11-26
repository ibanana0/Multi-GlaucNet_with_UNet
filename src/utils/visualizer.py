"""
Visualization and Qualitative Evaluation Module for MultiGlaucNet:
Plots original fundus image, predicted OD and Vessel masks,
masked inputs (I_OD and I_vessel), and detection probabilities.
"""

import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Menjamin eksekusi headless aman di container server
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


def save_gradcam_multibranch_plot(
    cases_data: List[Dict[str, Any]],
    save_path: str = "results/xai/gradcam_representative_cases.png"
):
    """
    Menyimpan plot multi-panel Grad-CAM (DPI 300) untuk kasus-kasus representatif (TP, TN, FP, FN).

    Setiap elemen cases_data harus berupa dict dengan kunci:
        - "category": str (misal "True Positive (Glaucoma)")
        - "image_id": str
        - "prob": float
        - "true_label": int
        - "I_orig": np.ndarray [H, W, 3] dalam [0, 1]
        - "I_od": np.ndarray [H, W, 3] dalam [0, 1]
        - "I_vessel": np.ndarray [H, W, 3] dalam [0, 1]
        - "cams": Dict[str, np.ndarray] dengan kunci "orig", "od", "vessel"
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(cases_data)
    if n_rows == 0:
        print("⚠️ Tidak ada data kasus untuk divisualisasikan pada Grad-CAM.")
        return

    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.6))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "Citra Asli (I_orig)",
        "Masked OD (I_OD)",
        "Masked Vessel (I_vessel)",
        "Grad-CAM: Global Retina",
        "Grad-CAM: Optic Disc",
        "Grad-CAM: Pembuluh Darah"
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=10)

    for row_idx, case in enumerate(cases_data):
        img_orig = case["I_orig"]
        img_od = case["I_od"]
        img_vessel = case["I_vessel"]
        cams = case["cams"]

        # Fungsi overlay heatmap JET
        def get_overlay(base_img, cam_mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(cam_mask, 0.0, 1.0)), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            overlay = 0.55 * base_img + 0.45 * heatmap
            return np.clip(overlay, 0.0, 1.0)

        overlay_orig = get_overlay(img_orig, cams.get("orig", np.zeros_like(img_orig[:, :, 0])))
        # Overlay Grad-CAM OD langsung pada img_od agar tetap selaras saat ROI-cropped
        overlay_od = get_overlay(img_od, cams.get("od", np.zeros_like(img_od[:, :, 0])))
        overlay_vessel = get_overlay(img_orig, cams.get("vessel", np.zeros_like(img_orig[:, :, 0])))

        # Tampilkan pada baris
        axes[row_idx, 0].imshow(img_orig)
        axes[row_idx, 1].imshow(img_od)
        axes[row_idx, 2].imshow(img_vessel)
        axes[row_idx, 3].imshow(overlay_orig)
        axes[row_idx, 4].imshow(overlay_od)
        axes[row_idx, 5].imshow(overlay_vessel)

        # Label kategori di sebelah kiri
        label_str = "Glaukoma" if case["true_label"] == 1 else "Normal"
        pred_str = "Glaukoma" if case["prob"] >= 0.5 else "Normal"
        vcdr_val = case.get("vcdr", None)
        vcdr_line = f"\nvCDR: {vcdr_val:.2f}" if (vcdr_val is not None and np.isfinite(vcdr_val)) else ""
        row_label = f"[{case['category']}]\nID: {case['image_id']}\nGT: {label_str}\nPred: {pred_str} (P={case['prob']:.2f}){vcdr_line}"
        axes[row_idx, 0].set_ylabel(row_label, fontsize=10, fontweight="semibold")

        for c in range(n_cols):
            axes[row_idx, c].set_xticks([])
            axes[row_idx, c].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Visualisasi Multi-Branch Grad-CAM disimpan ke: {save_path} (DPI 300)")


def save_branch_attribution_boxplot(
    df_attributions,
    save_path: str = "results/xai/branch_attribution_distribution.png"
):
    """
    Menyimpan plot distribusi kontribusi persentase 3 cabang untuk kelas Normal vs Glaukoma (DPI 300).
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["pct_orig", "pct_od", "pct_vessel"]
    labels = ["Cabang Citra Asli", "Cabang Optic Disc", "Cabang Pembuluh Darah"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    data_to_plot = []
    tick_positions = []
    tick_labels = []

    pos = 1
    for i, col in enumerate(categories):
        normal_data = df_attributions[df_attributions["true_label"] == 0][col].dropna().values
        glaucoma_data = df_attributions[df_attributions["true_label"] == 1][col].dropna().values

        data_to_plot.append(normal_data if len(normal_data) > 0 else [0.0])
        data_to_plot.append(glaucoma_data if len(glaucoma_data) > 0 else [0.0])

        tick_positions.append(pos + 0.5)
        tick_labels.append(labels[i])
        pos += 3

    positions = [1, 2, 4, 5, 7, 8]
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops=dict(color="black", linewidth=1.5),
        meanprops=dict(color="blue", linestyle="--", linewidth=1.5)
    )

    box_colors = ["#A1C9F4", "#FF9F9B", "#A1C9F4", "#FF9F9B", "#A1C9F4", "#FF9F9B"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=12, fontweight="semibold")
    ax.set_ylabel("Kontribusi Relatif (%)", fontsize=12, fontweight="semibold")
    ax.set_title("Distribusi Kontribusi Cabang ResNet-50 pada Test Set SMDG-19", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#A1C9F4", edgecolor="black", label="Mata Normal (Negatif)"),
        Patch(facecolor="#FF9F9B", edgecolor="black", label="Mata Glaukoma (Positif)"),
        plt.Line2D([0], [0], color="blue", linestyle="--", linewidth=1.5, label="Mean Kontribusi")
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Visualisasi Distribusi Kontribusi Cabang disimpan ke: {save_path} (DPI 300)")
