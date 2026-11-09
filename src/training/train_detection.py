"""
Stage 2 Training Script: Three-Branch ResNet50 Glaucoma Detection
with Focal Loss Ablation Study (Scenarios 1, 2, and 3)
under Two-Stage Decoupled Training strategy.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.models import VanillaDUNet, HybridDUNet, ThreeBranchResNet50
from src.losses import FocalLoss
from src.dataset.smdg_dataset import get_dataloaders
from src.utils.metrics import compute_classification_metrics


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_segmentation_models(variant: str, seg_dir: str, device: torch.device):
    """Memuat checkpoint modul segmentasi OD dan Pembuluh Darah lalu membekukan parameternya."""
    seg_path = Path(seg_dir)
    od_ckpt = seg_path / f"best_{variant}_od.pth"
    vessel_ckpt = seg_path / f"best_{variant}_vessel.pth"

    if variant == "vanilla":
        model_od = VanillaDUNet(in_channels=3, out_classes=1)
        model_vessel = VanillaDUNet(in_channels=3, out_classes=1)
    else:
        model_od = HybridDUNet(in_channels=3, out_classes=1, backbone="resnet50")
        model_vessel = HybridDUNet(in_channels=3, out_classes=1, backbone="resnet50")

    if od_ckpt.exists():
        print(f"📦 Memuat checkpoint OD: {od_ckpt}")
        ckpt = torch.load(od_ckpt, map_location=device)
        model_od.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"⚠️ Checkpoint OD tidak ditemukan di {od_ckpt}. Menggunakan bobot terinisialisasi.")

    if vessel_ckpt.exists():
        print(f"📦 Memuat checkpoint Vessel: {vessel_ckpt}")
        ckpt = torch.load(vessel_ckpt, map_location=device)
        model_vessel.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"⚠️ Checkpoint Vessel tidak ditemukan di {vessel_ckpt}. Menggunakan bobot terinisialisasi.")

    model_od = model_od.to(device).eval()
    model_vessel = model_vessel.to(device).eval()

    # Kunci (freeze) semua parameter segmentasi
    for p in model_od.parameters():
        p.requires_grad = False
    for p in model_vessel.parameters():
        p.requires_grad = False

    return model_od, model_vessel


def train_one_epoch_detector(detector, model_od, model_vessel, loader, optimizer, criterion, device):
    detector.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training Detector", leave=False):
        I_orig = batch["image"].to(device)
        targets = batch["label"].to(device)

        # Stage 1: Ekstraksi masker biner dan masking elementwise
        with torch.no_grad():
            prob_od = model_od(I_orig)
            prob_vessel = model_vessel(I_orig)

            # Binarisasi tegas POST-01 (threshold t = 0.5)
            M_od = (prob_od > 0.5).float()
            M_vessel = (prob_vessel > 0.5).float()

            # Perkalian elementwise POST-02
            I_od = I_orig * M_od
            I_vessel = I_orig * M_vessel

        # Stage 2: Three-Branch ResNet50
        optimizer.zero_grad()
        logits = detector(I_orig, I_od, I_vessel)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * I_orig.size(0)

    return total_loss / len(loader.dataset)


def validate_detector(detector, model_od, model_vessel, loader, criterion, device):
    detector.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation Detector", leave=False):
            I_orig = batch["image"].to(device)
            targets = batch["label"].to(device)

            prob_od = model_od(I_orig)
            prob_vessel = model_vessel(I_orig)

            M_od = (prob_od > 0.5).float()
            M_vessel = (prob_vessel > 0.5).float()

            I_od = I_orig * M_od
            I_vessel = I_orig * M_vessel

            logits = detector(I_orig, I_od, I_vessel)
            loss = criterion(logits, targets)
            total_loss += loss.item() * I_orig.size(0)

            # Hitung probabilitas kelas glaukoma (index 1)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    n_samples = len(loader.dataset)
    val_loss = total_loss / n_samples
    metrics = compute_classification_metrics(np.array(all_probs), np.array(all_targets))
    metrics["loss"] = val_loss
    return metrics


def train_detection(
    variant: str = "vanilla",
    scenario: int = 1,
    csv_path: str = "data/raw/smdg-19/metadata_smdg19.csv",
    root_dir: str = "data/raw/smdg-19",
    seg_dir: str = "checkpoints/segmentation",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 5e-5,
    pretrained: bool = True,
    apply_imagenet_norm: bool = False,
    device_pref: str = "auto",
    output_dir: str = "checkpoints/detection"
):
    device = get_device(device_pref)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Konfigurasi Skenario Ablation Loss
    if scenario == 1:
        sc_name = "Scenario1_Baseline_CE"
        gamma, alpha = 0.0, 1.0
    elif scenario == 2:
        sc_name = "Scenario2_Standard_Focal"
        gamma = 2.0
        # Hitung rasio empiris dari dataset
        df_temp = get_dataloaders(csv_path, root_dir, batch_size=batch_size)[3]
        n_norm = sum(df_temp["glaucoma_label"] == 0)
        n_glauc = max(sum(df_temp["glaucoma_label"] == 1), 1)
        alpha = float(n_norm) / float(n_glauc)
    elif scenario == 3:
        sc_name = "Scenario3_Extreme_MultiGlaucNet"
        gamma, alpha = 10.0, 3.0
    else:
        raise ValueError(f"Skenario tidak valid: {scenario}. Pilih 1, 2, atau 3.")

    alpha_str = f"{alpha:.2f}" if isinstance(alpha, (int, float)) else str(alpha)
    print(f"\n🚀 Menjalankan Pelatihan Deteksi Glaukoma:")
    print(f"   - Segmenter Variant : {variant.upper()}")
    print(f"   - Ablation Scenario : {sc_name}")
    print(f"   - Focal Loss Config : gamma={gamma}, alpha={alpha_str}")
    print(f"   - Device            : {device}")
    print(f"   - Epochs            : {epochs} | Batch Size: {batch_size} | LR: {lr}")

    # 2. Muat Data
    train_loader, val_loader, test_loader, _ = get_dataloaders(csv_path, root_dir, batch_size=batch_size)

    # 3. Muat Modul Segmentasi Beku
    model_od, model_vessel = load_segmentation_models(variant, seg_dir, device)

    # 4. Inisialisasi Detektor
    detector = ThreeBranchResNet50(
        num_classes=2,
        pretrained=pretrained,
        fusion_method="average",
        apply_imagenet_norm=apply_imagenet_norm
    ).to(device)

    # 5. Loss & Optimizer
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = optim.AdamW(detector.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 6. Training Loop
    best_auc = -1.0
    history = []
    tag = f"detector_{variant}_sc{scenario}"
    best_ckpt = output_path / f"best_{tag}.pth"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_detector(detector, model_od, model_vessel, train_loader, optimizer, criterion, device)
        val_metrics = validate_detector(detector, model_od, model_vessel, val_loader, criterion, device)
        scheduler.step()

        val_auc = val_metrics["auc_roc"]
        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            torch.save({
                "epoch": epoch,
                "variant": variant,
                "scenario": scenario,
                "sc_name": sc_name,
                "gamma": gamma,
                "alpha": alpha,
                "model_state_dict": detector.state_dict(),
                "best_auc": best_auc,
                "val_metrics": val_metrics
            }, best_ckpt)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_accuracy": val_acc,
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"]
        }
        history.append(epoch_log)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val Acc: {val_acc:.4f} "
            f"{'🌟 (Best Checkpoint Saved)' if is_best else ''}"
        )

    log_path = output_path / f"history_{tag}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    # 7. Evaluasi Independen pada Test Set menggunakan Checkpoint Terbaik
    print(f"\n🔍 Menjalankan Evaluasi Independen pada Test Set...")
    best_checkpoint = torch.load(best_ckpt, map_location=device)
    detector.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = validate_detector(detector, model_od, model_vessel, test_loader, criterion, device)

    test_log_path = output_path / f"test_metrics_{tag}.json"
    with open(test_log_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n✅ Pelatihan {tag.upper()} Selesai!")
    print(f"   Best Val AUC     : {best_auc:.4f}")
    print(f"   Test Set AUC     : {test_metrics['auc_roc']:.4f}")
    print(f"   Test Set Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test Set F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"   Test Sensitivity : {test_metrics['sensitivity']:.4f}")
    print(f"   Test Specificity : {test_metrics['specificity']:.4f}")
    print(f"   Checkpoint Path  : {best_ckpt}")
    print(f"   History Log      : {log_path}")
    print(f"   Test Metrics Log : {test_log_path}\n")

    return best_ckpt, history


def find_default_dataset_paths() -> tuple:
    candidates = [
        (PROJECT_ROOT.parent / "dataset" / "smdg-19" / "archive" / "metadata - standardized.csv", PROJECT_ROOT.parent / "dataset" / "smdg-19" / "archive"),
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
    parser = argparse.ArgumentParser(description="MultiGlaucNet Stage 2: Glaucoma Detection Training")
    parser.add_argument("--variant", choices=["vanilla", "hybrid"], default="vanilla")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], default=1, help="Skenario Loss: 1 (CE), 2 (Std Focal), 3 (Extreme)")
    parser.add_argument("--csv-path", default=default_csv)
    parser.add_argument("--root-dir", default=default_root)
    parser.add_argument("--seg-dir", default="checkpoints/segmentation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--apply-imagenet-norm", action="store_true", help="Aktifkan normalisasi ImageNet z-score")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output-dir", default="checkpoints/detection")
    args = parser.parse_args()

    train_detection(
        variant=args.variant,
        scenario=args.scenario,
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        seg_dir=args.seg_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        apply_imagenet_norm=args.apply_imagenet_norm,
        device_pref=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
