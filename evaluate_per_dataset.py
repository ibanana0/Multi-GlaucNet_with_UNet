"""
Script Evaluasi Terperinci Berdasarkan Sub-Dataset Asal (dataset_origin)
Menghitung rata-rata metrik per batch untuk segmentasi dan deteksi per sub-dataset pada SMDG-19 Test Set.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import VanillaDUNet, HybridDUNet, ThreeBranchResNet50
from src.dataset.smdg_dataset import get_dataloaders
from src.utils.metrics import compute_segmentation_metrics, compute_classification_metrics
from src.utils.roi_crop import crop_od_roi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = PROJECT_ROOT / "../dataset/smdg-19/archive/metadata - standardized.csv"
ROOT_DIR = PROJECT_ROOT / "../dataset/smdg-19/archive"
OUTPUT_DIR = PROJECT_ROOT / "results/per_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

split_df = pd.read_csv(PROJECT_ROOT / "splits/master_split_seed42.csv")
NAME_TO_ORIGIN = dict(zip(split_df["names"], split_df["dataset_origin"]))

def evaluate_segmentation_breakdown(variant: str, target: str, ckpt_path: str):
    print(f"\n🔍 [Segmentasi] Mengevaluasi {variant.upper()} - Target: {target.upper()}...")
    if variant == "vanilla":
        model = VanillaDUNet(in_channels=3, out_classes=1)
    else:
        model = HybridDUNet(in_channels=3, out_classes=1, backbone="resnet50")
        
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()

    _, _, test_loader, _ = get_dataloaders(
        csv_path=str(CSV_PATH),
        root_dir=str(ROOT_DIR),
        batch_size=4,
        target_filter=target
    )
    
    test_df = test_loader.dataset.df
    mask_key = f"{target}_mask"

    # Akumulasi metrik per origin
    origin_metrics = {}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Eval {variant}_{target}")):
            images = batch["image"].to(DEVICE)
            masks = batch[mask_key].to(DEVICE)
            b_size = images.size(0)
            
            # Hitung prediksi
            outputs = model(images)
            preds = outputs

            # Identifikasi origin dari batch ini melalui index dataset
            start_idx = i * 4
            batch_names = test_df.iloc[start_idx : start_idx + b_size]["names"].values

            # Evaluasi per sampel di batch
            for j in range(b_size):
                p_sample = preds[j:j+1]
                t_sample = masks[j:j+1]
                m = compute_segmentation_metrics(p_sample, t_sample)
                orig = NAME_TO_ORIGIN.get(batch_names[j], "OTHER")

                if orig not in origin_metrics:
                    origin_metrics[orig] = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0, "accuracy": 0.0, "count": 0}
                
                for k in ["dice", "iou", "sensitivity", "specificity", "accuracy"]:
                    origin_metrics[orig][k] += m[k]
                origin_metrics[orig]["count"] += 1

    results = {}
    for orig in sorted(origin_metrics.keys()):
        cnt = origin_metrics[orig]["count"]
        res = {k: origin_metrics[orig][k] / cnt for k in ["dice", "iou", "sensitivity", "specificity", "accuracy"]}
        res["sample_count"] = cnt
        results[orig] = res
        print(f"  > {orig:10s} ({cnt:3d} sampel): Dice={res['dice']:.4f}, IoU={res['iou']:.4f}, Sens={res['sensitivity']:.4f}, Spec={res['specificity']:.4f}")

    return results

def evaluate_detector_breakdown(tag: str, ckpt_path: str, seg_variant: str, seg_dir: str):
    print(f"\n🔍 [Deteksi] Mengevaluasi {tag}...")
    seg_path = Path(seg_dir)
    od_ckpt = seg_path / f"best_{seg_variant}_od.pth"
    vessel_ckpt = seg_path / f"best_{seg_variant}_vessel.pth"

    if seg_variant == "vanilla":
        model_od = VanillaDUNet(in_channels=3, out_classes=1)
        model_vessel = VanillaDUNet(in_channels=3, out_classes=1)
    else:
        model_od = HybridDUNet(in_channels=3, out_classes=1, backbone="resnet50")
        model_vessel = HybridDUNet(in_channels=3, out_classes=1, backbone="resnet50")

    if od_ckpt.exists():
        model_od.load_state_dict(torch.load(od_ckpt, map_location=DEVICE)["model_state_dict"])
    if vessel_ckpt.exists():
        model_vessel.load_state_dict(torch.load(vessel_ckpt, map_location=DEVICE)["model_state_dict"])

    model_od = model_od.to(DEVICE).eval()
    model_vessel = model_vessel.to(DEVICE).eval()

    detector = ThreeBranchResNet50(pretrained=False, num_classes=2, fusion_method="average")
    det_ckpt = torch.load(ckpt_path, map_location=DEVICE)
    detector.load_state_dict(det_ckpt["model_state_dict"])
    detector = detector.to(DEVICE).eval()
    use_od_roi_crop = bool(det_ckpt.get("use_od_roi_crop", False))

    _, _, test_loader, _ = get_dataloaders(
        csv_path=str(CSV_PATH),
        root_dir=str(ROOT_DIR),
        batch_size=4,
        target_filter=None
    )
    test_df = test_loader.dataset.df

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {tag}"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            m_od = (torch.sigmoid(model_od(images)) > 0.5).float()
            m_vessel = (torch.sigmoid(model_vessel(images)) > 0.5).float()

            i_od = crop_od_roi(images, m_od) if use_od_roi_crop else (images * m_od)
            i_vessel = images * m_vessel

            logits = detector(images, i_od, i_vessel)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels_np = labels.cpu().numpy()

            for p, l in zip(probs, labels_np):
                all_probs.append(p)
                all_labels.append(l)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    dataset_origins = np.array([NAME_TO_ORIGIN.get(n, "OTHER") for n in test_df["names"]])

    results = {}
    unique_origins = sorted(list(set(dataset_origins)))
    for ds in unique_origins:
        idx = np.where(dataset_origins == ds)[0]
        if len(idx) == 0:
            continue
        ds_probs = all_probs[idx]
        ds_labels = all_labels[idx]
        
        ds_metrics = compute_classification_metrics(ds_probs, ds_labels)
        ds_metrics["sample_count"] = int(len(idx))
        ds_metrics["normal_count"] = int(np.sum(ds_labels == 0))
        ds_metrics["glaucoma_count"] = int(np.sum(ds_labels == 1))
        results[ds] = ds_metrics
        
        auc_val = ds_metrics.get("auc_roc")
        auc_str = f"{auc_val:.4f}" if auc_val is not None and not np.isnan(auc_val) else "N/A"
        print(f"  > {ds:10s} ({len(idx):3d} sampel | N:{ds_metrics['normal_count']:3d} G:{ds_metrics['glaucoma_count']:3d}): Acc={ds_metrics['accuracy']:.4f}, Sens={ds_metrics['sensitivity']:.4f}, Spec={ds_metrics['specificity']:.4f}, AUC={auc_str}")

    return results

def main():
    final_report = {"segmentation": {}, "detection": {}}

    # 1. Segmentasi
    seg_ckpts = {
        ("vanilla", "od"): "checkpoints/segmentation/best_vanilla_od.pth",
        ("hybrid", "od"): "checkpoints/segmentation/best_hybrid_od.pth",
        ("hybrid", "oc"): "checkpoints/segmentation/best_hybrid_oc.pth",
        ("vanilla", "vessel"): "checkpoints/segmentation/best_vanilla_vessel.pth",
        ("hybrid", "vessel"): "checkpoints/segmentation/best_hybrid_vessel.pth",
    }
    for (var, tgt), ckpt in seg_ckpts.items():
        res = evaluate_segmentation_breakdown(var, tgt, str(PROJECT_ROOT / ckpt))
        final_report["segmentation"][f"{var}_{tgt}"] = res

    # 2. Deteksi
    det_configs = [
        ("Hybrid_Scenario2_Proposed", "checkpoints/detection/best_detector_hybrid_sc2.pth", "hybrid", "checkpoints/segmentation"),
        ("Hybrid_Scenario3_ExtremeFocal", "checkpoints/detection/best_detector_hybrid_sc3.pth", "hybrid", "checkpoints/segmentation"),
        ("Vanilla_Scenario1_CE", "checkpoints/detection/best_detector_vanilla_sc1.pth", "vanilla", "checkpoints/segmentation"),
    ]
    for tag, ckpt, seg_var, seg_dir in det_configs:
        res = evaluate_detector_breakdown(tag, str(PROJECT_ROOT / ckpt), seg_var, str(PROJECT_ROOT / seg_dir))
        final_report["detection"][tag] = res

    # Simpan JSON
    json_path = OUTPUT_DIR / "per_dataset_summary.json"
    with open(json_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n✅ Berhasil menyimpan laporan breakdown ke {json_path}")

if __name__ == "__main__":
    main()
