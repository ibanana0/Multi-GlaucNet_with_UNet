#!/usr/bin/env bash
# ==============================================================================
# MultiGlaucNet: Phase 1 - Track 1 Head-to-Head Comparison Experiment (REFUGE)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "🚀 MEMULAI EKSPERIMEN PHASE 1: HEAD-TO-HEAD REFUGE COMPARISON"
echo "Waktu Mulai: $(date)"
echo "GPU Device : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================================================"

DATASET_CSV="../dataset/smdg-19/archive/metadata - standardized.csv"
DATASET_ROOT="../dataset/smdg-19/archive"
OUT_DIR="checkpoints/h2h"
mkdir -p "$OUT_DIR"
mkdir -p "results/h2h"

# ------------------------------------------------------------------------------
# 1. SEGMENTASI OPTIC DISC (OD) PADA REFUGE
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "📌 [Task 1.1] Pelatihan Vanilla DUNet - Optic Disc (REFUGE)"
echo "========================================================================"
python3 src/training/train_segmentation.py \
    --variant vanilla \
    --target od \
    --subset refuge \
    --epochs 25 \
    --batch-size 4 \
    --device cuda \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT" \
    --output-dir "$OUT_DIR"

echo ""
echo "========================================================================"
echo "📌 [Task 1.2] Pelatihan Hybrid DUNet - Optic Disc (REFUGE - Proposed Novelty)"
echo "========================================================================"
python3 src/training/train_segmentation.py \
    --variant hybrid \
    --target od \
    --subset refuge \
    --epochs 25 \
    --batch-size 4 \
    --device cuda \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT" \
    --output-dir "$OUT_DIR"

# ------------------------------------------------------------------------------
# 2. SEGMENTASI OPTIC CUP (OC) PADA REFUGE
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "📌 [Task 1.3] Pelatihan Hybrid DUNet - Optic Cup (REFUGE - CDR Biomarker)"
echo "========================================================================"
python3 src/training/train_segmentation.py \
    --variant hybrid \
    --target oc \
    --subset refuge \
    --epochs 25 \
    --batch-size 4 \
    --device cuda \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT" \
    --output-dir "$OUT_DIR"

# ------------------------------------------------------------------------------
# 3. SEGMENTASI PEMBULUH DARAH (VESSEL) UNTUK MEMANDU CABANG DETEKSI
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "📌 [Task 1.3b] Pelatihan Segmentasi Pembuluh Darah (Vessel Guard)"
echo "========================================================================"
python3 src/training/train_segmentation.py \
    --variant vanilla \
    --target vessel \
    --epochs 25 \
    --batch-size 4 \
    --device cuda \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT" \
    --output-dir "$OUT_DIR"

python3 src/training/train_segmentation.py \
    --variant hybrid \
    --target vessel \
    --epochs 25 \
    --batch-size 4 \
    --device cuda \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT" \
    --output-dir "$OUT_DIR"

# ------------------------------------------------------------------------------
# 4. DETEKSI GLAUKOMA TIGA CABANG PADA REFUGE (SKENARIO 3: EXTREME FOCAL LOSS)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "📌 [Task 1.4] Pelatihan Tiga Cabang ResNet50 + Vanilla DUNet (REFUGE)"
echo "========================================================================"
python3 src/training/train_detection.py \
    --variant vanilla \
    --scenario 3 \
    --subset refuge \
    --epochs 20 \
    --batch-size 4 \
    --device cuda \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT" \
    --seg-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR"

echo ""
echo "========================================================================"
echo "📌 [Task 1.5] Pelatihan Tiga Cabang ResNet50 + Hybrid DUNet (REFUGE - Proposed)"
echo "========================================================================"
python3 src/training/train_detection.py \
    --variant hybrid \
    --scenario 3 \
    --subset refuge \
    --epochs 20 \
    --batch-size 4 \
    --device cuda \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT" \
    --seg-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR"

echo ""
echo "========================================================================"
echo "🎉 PHASE 1 (HEAD-TO-HEAD REFUGE) SELESAI DENGAN SUKSES!"
echo "Waktu Selesai: $(date)"
echo "========================================================================"
