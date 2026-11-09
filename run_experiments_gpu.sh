#!/usr/bin/env bash
# ==============================================================================
# MultiGlaucNet: Full Experiment Suite (GPU Version)
# Menjalankan seluruh alur eksperimen:
# 1. Segmentasi OD & Vessel (Vanilla DUNet & Hybrid DUNet)
# 2. Deteksi Glaukoma Tiga Cabang (Ablation Study Skenario 1, 2, dan 3)
# ==============================================================================

set -e  # Hentikan eksekusi jika terjadi error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "🚀 MEMULAI EKSPERIMEN LENGKAP MULTIGLAUCNET (MODE GPU)"
echo "Waktu Mulai: $(date)"
echo "========================================================================"

# Path konfigurasi dataset
DATASET_CSV="../dataset/smdg-19/archive/metadata - standardized.csv"
DATASET_ROOT="../dataset/smdg-19/archive"

DEVICE="cuda"
SEG_EPOCHS=25
DET_EPOCHS=20
BATCH_SIZE=4

# ------------------------------------------------------------------------------
# TAHAP 1: PELATIHAN SEGMENTASI (STAGE 1)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "📌 TAHAP 1A: Pelatihan Segmentasi Vanilla DUNet (OD & Vessel)"
echo "========================================================================"
python3 src/training/train_segmentation.py \
    --variant vanilla \
    --target both \
    --epochs $SEG_EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT"

echo ""
echo "========================================================================"
echo "📌 TAHAP 1B: Pelatihan Segmentasi Hybrid DUNet (OD & Vessel - Novelty)"
echo "========================================================================"
python3 src/training/train_segmentation.py \
    --variant hybrid \
    --target both \
    --epochs $SEG_EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT"

# ------------------------------------------------------------------------------
# TAHAP 2: PELATIHAN DETEKSI TIGA CABANG & ABLATION STUDY (STAGE 2)
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "📌 TAHAP 2A: Deteksi Glaukoma (Vanilla DUNet + Skenario 1: Cross-Entropy)"
echo "========================================================================"
python3 src/training/train_detection.py \
    --variant vanilla \
    --scenario 1 \
    --epochs $DET_EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT"

echo ""
echo "========================================================================"
echo "📌 TAHAP 2B: Deteksi Glaukoma (Vanilla DUNet + Skenario 2: Standard Focal Loss)"
echo "========================================================================"
python3 src/training/train_detection.py \
    --variant vanilla \
    --scenario 2 \
    --epochs $DET_EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT"

echo ""
echo "========================================================================"
echo "📌 TAHAP 2C: Deteksi Glaukoma (Vanilla DUNet + Skenario 3: Extreme Focal Loss)"
echo "========================================================================"
python3 src/training/train_detection.py \
    --variant vanilla \
    --scenario 3 \
    --epochs $DET_EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT"

echo ""
echo "========================================================================"
echo "📌 TAHAP 2D: Deteksi Glaukoma (Hybrid DUNet + Skenario 2: Standard Focal Loss)"
echo "========================================================================"
python3 src/training/train_detection.py \
    --variant hybrid \
    --scenario 2 \
    --epochs $DET_EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT"

echo ""
echo "========================================================================"
echo "📌 TAHAP 2E: Deteksi Glaukoma (Hybrid DUNet + Skenario 1 & 3 untuk komparasi penuh)"
echo "========================================================================"
python3 src/training/train_detection.py \
    --variant hybrid \
    --scenario 1 \
    --epochs $DET_EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT"

python3 src/training/train_detection.py \
    --variant hybrid \
    --scenario 3 \
    --epochs $DET_EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT"

# ------------------------------------------------------------------------------
# TAHAP 3: EVALUASI LENGKAP PADA TEST SET & PEMBUATAN TABEL LAPORAN BAB 4
# ------------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "📌 TAHAP 3: Evaluasi Lengkap Test Set & Pembuatan Tabel Laporan Bab 4"
echo "========================================================================"
python3 evaluate_test_set.py \
    --device $DEVICE \
    --csv-path "$DATASET_CSV" \
    --root-dir "$DATASET_ROOT"

echo ""
echo "========================================================================"
echo "🎉 SELURUH EKSPERIMEN SELESAI DENGAN SUKSES!"
echo "Hasil evaluasi dan log riwayat tersimpan di folder: results/"
echo "Visualisasi hasil prediksi tersimpan di folder    : results/visualizations/"
echo "Waktu Selesai: $(date)"
echo "========================================================================"
