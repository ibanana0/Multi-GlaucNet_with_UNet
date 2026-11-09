#!/usr/bin/env bash
# ==============================================================================
# MultiGlaucNet: Full Experiment Suite (CPU Version)
# Dirancang untuk eksekusi pada CPU (pengujian fungsionalitas / resource terbatas).
# Parameter epoch dan batch size disesuaikan agar tidak membebani komputasi CPU.
# ==============================================================================

set -e  # Hentikan eksekusi jika terjadi error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Aktifkan virtual environment jika ada
if [ -f "/root/venv/bin/activate" ]; then
    source /root/venv/bin/activate
fi

echo "========================================================================"
echo "🖥️ MEMULAI EKSPERIMEN MULTIGLAUCNET (MODE CPU)"
echo "Waktu Mulai: $(date)"
echo "Catatan: Deformable Conv 512x512 pada CPU membutuhkan waktu lebih lama."
echo "========================================================================"

# Path konfigurasi dataset
DATASET_CSV="../dataset/smdg-19/archive/metadata - standardized.csv"
DATASET_ROOT="../dataset/smdg-19/archive"

DEVICE="cpu"
# Nilai default hemat CPU (dapat dioverride dengan export SEG_EPOCHS=X / DET_EPOCHS=X)
SEG_EPOCHS=${SEG_EPOCHS:-2}
DET_EPOCHS=${DET_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}

echo "Konfigurasi: Device=$DEVICE | Seg Epochs=$SEG_EPOCHS | Det Epochs=$DET_EPOCHS | Batch Size=$BATCH_SIZE"

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
echo "📌 TAHAP 1B: Pelatihan Segmentasi Hybrid DUNet (OD & Vessel - Novelty CBAM + Focal Tversky)"
echo "========================================================================"
python3 src/training/train_segmentation.py \
    --variant hybrid \
    --target both \
    --loss focal_tversky \
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
echo "🎉 SELURUH EKSPERIMEN CPU SELESAI DENGAN SUKSES!"
echo "Hasil evaluasi dan log riwayat tersimpan di folder: results/"
echo "Visualisasi hasil prediksi tersimpan di folder    : results/visualizations/"
echo "Waktu Selesai: $(date)"
echo "========================================================================"
