"""
Comprehensive Test Set Evaluation & Benchmarking Script for MultiGlaucNet:
1. Evaluates Stage 1 Segmentation Models (Vanilla vs Hybrid DUNet for OD and Vessel) on unseen Test Set.
2. Evaluates Stage 2 Glaucoma Detection Models across Ablation Scenarios (CE, Standard Focal, Extreme Focal).
3. Computes complete clinical metrics: Dice, IoU, Accuracy, Sensitivity, Specificity, Precision, F1-Score, AUC-ROC.
4. Generates publication-ready figures (DPI 300):
   - Multi-scenario ROC curve comparison
   - Qualitative Contour Overlay on fundus images
5. Exports structured markdown and JSON tables for Chapter 4 Report.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.models import VanillaDUNet, HybridDUNet, ThreeBranchResNet50
from src.dataset.smdg_dataset import get_dataloaders
from src.utils.metrics import compute_segmentation_metrics, compute_classification_metrics
from src.utils.visualizer import (
    save_segmentation_prediction_plot,
    save_contour_overlay_plot,
    plot_multimodel_roc_curve,
    save_three_branch_overview_plot
)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_segmentation_model(model_variant: str, target: str, ckpt_path: Path, csv_path: str, root_dir: str, device: torch.device, batch_size: int = 4):
    """Mengevaluasi satu model segmentasi pada test set independen."""
    print(f"\n🔍 Menguji Segmentasi: {model_variant.upper()} - {target.upper()}...")
    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint tidak ditemukan di: {ckpt_path}. Melewati evaluasi.")
        return None

    # Muat Test DataLoader dengan Target Filter
    _, _, test_loader, _ = get_dataloaders(
        csv_path=csv_path,
        root_dir=root_dir,
        batch_size=batch_size,
        target_filter=target
    )

    if model_variant == "vanilla":
        model = VanillaDUNet(in_channels=3, out_classes=1)
    else:
        model = HybridDUNet(in_channels=3, out_classes=1, backbone="resnet50")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    target_key = "od_mask" if target == "od" else "vessel_mask"
    metric_accum = {"dice": 0.0, "iou": 0.0, "accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    n_samples = 0

    first_batch_saved = False

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {target.upper()}", leave=False):
            images = batch["image"].to(device)
            targets = batch[target_key].to(device)

            preds = model(images)
            b_size = images.size(0)
            n_samples += b_size

            m = compute_segmentation_metrics(preds, targets)
            for k in metric_accum:
                metric_accum[k] += m[k] * b_size

            # Simpan satu sampel visualisasi pertama
            if not first_batch_saved and b_size > 0:
                sample_img = images[0]
                sample_pred = preds[0, 0]
                sample_gt = targets[0, 0]
                vis_path = PROJECT_ROOT / f"results/visualizations/test_sample_seg_{model_variant}_{target}.png"
                save_segmentation_prediction_plot(
                    fundus_img=sample_img,
                    pred_od=sample_pred if target == "od" else torch.zeros_like(sample_pred),
                    pred_vessel=sample_pred if target == "vessel" else torch.zeros_like(sample_pred),
                    gt_od=sample_gt if target == "od" else None,
                    gt_vessel=sample_gt if target == "vessel" else None,
                    save_path=str(vis_path)
                )
                first_batch_saved = True

    final_metrics = {k: v / max(n_samples, 1) for k, v in metric_accum.items()}
    final_metrics["test_samples"] = n_samples
    print(f"   ✅ Hasil Test Set {model_variant.upper()} - {target.upper()}:")
    print(f"      Dice: {final_metrics['dice']:.4f} | IoU: {final_metrics['iou']:.4f} | "
          f"Sens: {final_metrics['sensitivity']:.4f} | Spec: {final_metrics['specificity']:.4f} | "
          f"Acc: {final_metrics['accuracy']:.4f}")
    return final_metrics


def evaluate_detector_model(
    detector_tag: str,
    ckpt_path: Path,
    seg_variant: str,
    seg_dir: Path,
    csv_path: str,
    root_dir: str,
    device: torch.device,
    batch_size: int = 4
):
    """Mengevaluasi satu varian detektor glaukoma pada test set independen."""
    print(f"\n🔍 Menguji Detektor: {detector_tag}...")
    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint tidak ditemukan di: {ckpt_path}. Melewati evaluasi.")
        return None

    # Muat Test DataLoader (tanpa filter organ karena menguji klasifikasi glaukoma)
    _, _, test_loader, _ = get_dataloaders(
        csv_path=csv_path,
        root_dir=root_dir,
        batch_size=batch_size,
        target_filter=None
    )

    # 1. Muat Segmentasi OD & Vessel
    od_ckpt = seg_dir / f"best_{seg_variant}_od.pth"
    vessel_ckpt = seg_dir / f"best_{seg_variant}_vessel.pth"

    if seg_variant == "vanilla":
        model_od = VanillaDUNet(in_channels=3, out_classes=1)
        model_vessel = VanillaDUNet(in_channels=3, out_classes=1)
    else:
        model_od = HybridDUNet(in_channels=3, out_classes=1, backbone="resnet50")
        model_vessel = HybridDUNet(in_channels=3, out_classes=1, backbone="resnet50")

    if od_ckpt.exists():
        model_od.load_state_dict(torch.load(od_ckpt, map_location=device)["model_state_dict"])
    if vessel_ckpt.exists():
        model_vessel.load_state_dict(torch.load(vessel_ckpt, map_location=device)["model_state_dict"])

    model_od = model_od.to(device).eval()
    model_vessel = model_vessel.to(device).eval()

    # 2. Muat Three-Branch ResNet-50 Detector
    detector = ThreeBranchResNet50(num_classes=2, pretrained=False, fusion_method="average")
    det_ckpt = torch.load(ckpt_path, map_location=device)
    detector.load_state_dict(det_ckpt["model_state_dict"])
    detector = detector.to(device).eval()

    all_probs = []
    all_targets = []
    saved_overlay = False

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {detector_tag}", leave=False):
            I_orig = batch["image"].to(device)
            targets = batch["label"].to(device)

            prob_od = model_od(I_orig)
            prob_vessel = model_vessel(I_orig)

            M_od = (prob_od > 0.5).float()
            M_vessel = (prob_vessel > 0.5).float()

            I_od = I_orig * M_od
            I_vessel = I_orig * M_vessel

            logits = detector(I_orig, I_od, I_vessel)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

            # Simpan satu plot contour overlay beresolusi DPI 300
            if not saved_overlay and I_orig.size(0) > 0:
                save_contour_overlay_plot(
                    fundus_img=I_orig[0],
                    pred_od=M_od[0, 0],
                    pred_vessel=M_vessel[0, 0],
                    save_path=str(PROJECT_ROOT / f"results/visualizations/contour_overlay_{detector_tag}.png")
                )
                save_three_branch_overview_plot(
                    I_orig=I_orig[0],
                    I_od=I_od[0],
                    I_vessel=I_vessel[0],
                    pred_label="Glaukoma" if probs[0] > 0.5 else "Normal",
                    confidence=float(probs[0] if probs[0] > 0.5 else (1.0 - probs[0])),
                    true_label="Glaukoma" if targets[0] == 1 else "Normal",
                    save_path=str(PROJECT_ROOT / f"results/visualizations/three_branch_{detector_tag}.png")
                )
                saved_overlay = True

    metrics = compute_classification_metrics(
        np.array(all_probs),
        np.array(all_targets),
        threshold=0.5,
        return_roc_curve=True
    )

    print(f"   ✅ Hasil Test Set {detector_tag}:")
    print(f"      AUC-ROC: {metrics['auc_roc']:.4f} | Accuracy: {metrics['accuracy']:.4f} | "
          f"F1-Score: {metrics['f1_score']:.4f} | Sensitivity: {metrics['sensitivity']:.4f} | "
          f"Specificity: {metrics['specificity']:.4f} | Precision: {metrics['precision']:.4f}")
    return metrics


def find_default_dataset_paths() -> tuple:
    """Mencari lokasi dataset SMDG-19 yang tersedia secara otomatis."""
    candidates = [
        (PROJECT_ROOT.parent / "dataset" / "metadata - standardized.csv", PROJECT_ROOT.parent / "dataset"),
        (PROJECT_ROOT / "dataset" / "metadata - standardized.csv", PROJECT_ROOT / "dataset"),
        (PROJECT_ROOT / "data/raw/smdg-19/metadata_smdg19.csv", PROJECT_ROOT / "data/raw/smdg-19"),
    ]
    for csv_c, root_c in candidates:
        if csv_c.exists():
            return str(csv_c), str(root_c)
    return str(candidates[0][0]), str(candidates[0][1])


def main():
    default_csv, default_root = find_default_dataset_paths()
    parser = argparse.ArgumentParser(description="MultiGlaucNet Full Test Set Evaluation Pipeline")
    parser.add_argument("--csv-path", default=default_csv, help="Path metadata CSV")
    parser.add_argument("--root-dir", default=default_root, help="Root direktori citra")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Komputasi device")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size evaluasi")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"🚀 Memulai Pengujian Independen Test Set MultiGlaucNet pada Device: {device}")

    results_dir = PROJECT_ROOT / "results"
    vis_dir = results_dir / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    seg_dir = PROJECT_ROOT / "checkpoints/segmentation"
    det_dir = PROJECT_ROOT / "checkpoints/detection"

    summary: Dict[str, Any] = {"segmentation": {}, "detection": {}}

    # =========================================================================
    # 1. EVALUASI TAHAP 1: SEGMENTASI (OD & VESSEL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TAHAP 1: EVALUASI SEGMENTASI OPTIC DISC & PEMBULUH DARAH PADA TEST SET")
    print("=" * 70)

    seg_models = [
        ("vanilla", "od", seg_dir / "best_vanilla_od.pth"),
        ("hybrid", "od", seg_dir / "best_hybrid_od.pth"),
        ("vanilla", "vessel", seg_dir / "best_vanilla_vessel.pth"),
        ("hybrid", "vessel", seg_dir / "best_hybrid_vessel.pth"),
    ]

    for variant, target, ckpt in seg_models:
        key = f"{variant}_{target}"
        res = evaluate_segmentation_model(variant, target, ckpt, args.csv_path, args.root_dir, device, args.batch_size)
        if res:
            summary["segmentation"][key] = res

    # =========================================================================
    # 2. EVALUASI TAHAP 2: DETEKSI GLAUKOMA & ABLASI FOCAL LOSS
    # =========================================================================
    print("\n" + "=" * 70)
    print("TAHAP 2: EVALUASI DETEKSI GLAUKOMA & ABLASI LOSS PADA TEST SET")
    print("=" * 70)

    detectors = [
        ("Vanilla_Scenario1_CE", det_dir / "best_detector_vanilla_sc1.pth", "vanilla", "#2ca02c", "--"),
        ("Vanilla_Scenario2_StdFocal", det_dir / "best_detector_vanilla_sc2.pth", "vanilla", "#1f77b4", "-."),
        ("Vanilla_Scenario3_ExtremeFocal", det_dir / "best_detector_vanilla_sc3.pth", "vanilla", "#ff7f0e", ":"),
        ("Hybrid_Scenario2_Proposed", det_dir / "best_detector_hybrid_sc2.pth", "hybrid", "#d62728", "-"),
    ]

    roc_plot_dict = {}

    for tag, ckpt, seg_var, color, line_style in detectors:
        res = evaluate_detector_model(tag, ckpt, seg_var, seg_dir, args.csv_path, args.root_dir, device, args.batch_size)
        if res:
            summary["detection"][tag] = {k: v for k, v in res.items() if k != "roc_curve"}
            if "roc_curve" in res and len(res["roc_curve"]["fpr"]) > 0:
                roc_plot_dict[tag] = {
                    "fpr": res["roc_curve"]["fpr"],
                    "tpr": res["roc_curve"]["tpr"],
                    "auc": res["auc_roc"],
                    "color": color,
                    "linestyle": line_style
                }

    # =========================================================================
    # 3. GENERATE PLOT KURVA ROC MULTI-MODEL (DPI 300)
    # =========================================================================
    if len(roc_plot_dict) > 0:
        roc_save_path = vis_dir / "multimodel_roc_curve.png"
        plot_multimodel_roc_curve(
            roc_dict=roc_plot_dict,
            save_path=str(roc_save_path),
            title="Kurva ROC Uji Deteksi Glaukoma pada SMDG-19 Test Set"
        )

    # =========================================================================
    # 4. EKSPOR HASIL DALAM FORMAT JSON & TABEL MARKDOWN
    # =========================================================================
    json_summary_path = results_dir / "test_evaluation_summary.json"
    with open(json_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Format Tabel Markdown untuk Bab 4
    md_content = ["# Rangkuman Pengujian Independen (SMDG-19 Test Set)", ""]
    md_content.append("## 1. Kinerja Segmentasi Tahap 1 (Optic Disc & Pembuluh Darah)")
    md_content.append("| Model Variant | Target Organ | Dice (F1) | IoU (Jaccard) | Recall | Specificity | Accuracy |")
    md_content.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for k, v in summary["segmentation"].items():
        var_name = "Vanilla DUNet" if "vanilla" in k else "Hybrid DUNet"
        tgt_name = "Optic Disc" if "od" in k else "Blood Vessel"
        md_content.append(
            f"| {var_name} | {tgt_name} | **{v['dice']:.4f}** | {v['iou']:.4f} | "
            f"{v['sensitivity']:.4f} | {v['specificity']:.4f} | {v['accuracy']:.4f} |"
        )

    md_content.append("\n## 2. Kinerja Klasifikasi Tahap 2 & Studi Ablasi Loss")
    md_content.append("| Konfigurasi Detektor | Loss Function | AUC-ROC | Accuracy | Precision | Recall | F1-Score | Specificity |")
    md_content.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for k, v in summary["detection"].items():
        md_content.append(
            f"| {k} | Multi-GlaucNet | **{v['auc_roc']:.4f}** | {v['accuracy']:.4f} | "
            f"{v['precision']:.4f} | {v['sensitivity']:.4f} | {v['f1_score']:.4f} | {v['specificity']:.4f} |"
        )

    md_table_path = results_dir / "test_evaluation_table.md"
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    print("\n" + "=" * 70)
    print("🎉 SELURUH PENGUJIAN TEST SET SELESAI DENGAN SUKSES!")
    print(f"   - JSON Summary   : {json_summary_path}")
    print(f"   - Markdown Table : {md_table_path}")
    print(f"   - Visualizations : {vis_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
