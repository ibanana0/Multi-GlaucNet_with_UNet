"""
Stage 1 Training Script: Optic Disc and Blood Vessel Segmentation
Supports training Vanilla DUNet and Hybrid DUNet on SMDG-19 dataset.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Tambahkan root direktori proyek ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Aktifkan fallback MPS untuk operator kustom seperti deform_conv2d pada Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import VanillaDUNet, HybridDUNet
from src.losses import BCEDiceLoss, DiceLoss
from src.dataset.smdg_dataset import get_dataloaders
from src.utils.metrics import compute_segmentation_metrics


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, target_key, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        targets = batch[target_key].to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, target_key, device):
    model.eval()
    total_loss = 0.0
    metric_accum = {"dice": 0.0, "iou": 0.0, "accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            targets = batch[target_key].to(device)

            preds = model(images)
            loss = criterion(preds, targets)
            total_loss += loss.item() * images.size(0)

            # Hitung metrik batch
            m = compute_segmentation_metrics(preds, targets)
            for k in metric_accum:
                metric_accum[k] += m[k] * images.size(0)

    n_samples = len(loader.dataset)
    val_loss = total_loss / n_samples
    val_metrics = {k: v / n_samples for k, v in metric_accum.items()}
    val_metrics["loss"] = val_loss
    return val_metrics


def train_segmentation(
    variant: str = "vanilla",
    target: str = "od",
    csv_path: str = "data/raw/smdg-19/metadata_smdg19.csv",
    root_dir: str = "data/raw/smdg-19",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-4,
    loss_type: str = "bce_dice",
    device_pref: str = "auto",
    output_dir: str = "checkpoints/segmentation"
):
    device = get_device(device_pref)
    print(f"🚀 Menjalankan Pelatihan Segmentasi:")
    print(f"   - Model Variant : {variant.upper()}")
    print(f"   - Target Organ  : {target.upper()} ({'Optic Disc' if target == 'od' else 'Blood Vessel'})")
    print(f"   - Device        : {device}")
    print(f"   - Epochs        : {epochs} | Batch Size: {batch_size} | LR: {lr}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Siapkan Dataloader dengan Target Filter (Sparsity Guard)
    target_key = "od_mask" if target == "od" else "vessel_mask"
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        csv_path=csv_path,
        root_dir=root_dir,
        batch_size=batch_size,
        target_filter=target
    )

    # 2. Inisialisasi Model
    if variant.lower() == "vanilla":
        model = VanillaDUNet(in_channels=3, out_classes=1)
    elif variant.lower() == "hybrid":
        model = HybridDUNet(in_channels=3, out_classes=1, backbone="resnet50")
    else:
        raise ValueError(f"Variant tidak dikenal: {variant}. Gunakan 'vanilla' atau 'hybrid'.")

    model = model.to(device)

    # 3. Loss & Optimizer
    criterion = BCEDiceLoss() if loss_type == "bce_dice" else DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Training Loop
    best_dice = -1.0
    history = []

    model_tag = f"{variant}_{target}"
    best_ckpt_path = output_path / f"best_{model_tag}.pth"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, target_key, device)
        val_metrics = validate(model, val_loader, criterion, target_key, device)
        scheduler.step()

        val_dice = val_metrics["dice"]
        val_loss = val_metrics["loss"]

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "variant": variant,
                "target": target,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "val_metrics": val_metrics
            }, best_ckpt_path)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_metrics["iou"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "lr": scheduler.get_last_lr()[0]
        }
        history.append(epoch_log)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} "
            f"{'🌟 (Best Checkpoint Saved)' if is_best else ''}"
        )

    # Simpan riwayat pelatihan
    log_path = output_path / f"history_{model_tag}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    # 5. Evaluasi Independen pada Test Set menggunakan Checkpoint Terbaik
    print(f"\n🔍 Menjalankan Evaluasi Independen pada Test Set...")
    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = validate(model, test_loader, criterion, target_key, device)

    test_log_path = output_path / f"test_metrics_{model_tag}.json"
    with open(test_log_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n✅ Pelatihan {model_tag.upper()} Selesai!")
    print(f"   Best Val Dice    : {best_dice:.4f}")
    print(f"   Test Set Dice    : {test_metrics['dice']:.4f}")
    print(f"   Test Set IoU     : {test_metrics['iou']:.4f}")
    print(f"   Test Sensitivity : {test_metrics['sensitivity']:.4f}")
    print(f"   Test Specificity : {test_metrics['specificity']:.4f}")
    print(f"   Test Accuracy    : {test_metrics['accuracy']:.4f}")
    print(f"   Checkpoint Path  : {best_ckpt_path}")
    print(f"   History Log      : {log_path}")
    print(f"   Test Metrics Log : {test_log_path}\n")

    return best_ckpt_path, history


def find_default_dataset_paths() -> tuple:
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
    parser = argparse.ArgumentParser(description="MultiGlaucNet Stage 1: Segmentation Training")
    parser.add_argument("--variant", choices=["vanilla", "hybrid"], default="vanilla", help="Model variant: vanilla atau hybrid")
    parser.add_argument("--target", choices=["od", "vessel", "both"], default="od", help="Target organ: od, vessel, atau both")
    parser.add_argument("--csv-path", default=default_csv, help="Path ke file CSV metadata")
    parser.add_argument("--root-dir", default=default_root, help="Root direktori citra")
    parser.add_argument("--epochs", type=int, default=15, help="Jumlah epoch pelatihan")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--loss", choices=["bce_dice", "dice"], default="bce_dice", help="Loss function")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Device target")
    parser.add_argument("--output-dir", default="checkpoints/segmentation", help="Folder simpan checkpoint")
    args = parser.parse_args()

    targets = ["od", "vessel"] if args.target == "both" else [args.target]

    for tgt in targets:
        print(f"\n{'='*60}")
        print(f"MEMULAI PELATIHAN: {args.variant.upper()} - {tgt.upper()}")
        print(f"{'='*60}")
        train_segmentation(
            variant=args.variant,
            target=tgt,
            csv_path=args.csv_path,
            root_dir=args.root_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            loss_type=args.loss,
            device_pref=args.device,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
