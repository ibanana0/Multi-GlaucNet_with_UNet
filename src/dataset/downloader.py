"""
Downloader & Dataset Manager for SMDG-19 (Standardized Multi-Channel Dataset for Glaucoma).
Supports Kaggle API, Kagglehub, local ZIP extraction, and synthetic sample generation.
"""

import os
import sys
import json
import zipfile
import argparse
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


class SMDGDownloader:
    """Mengelola pengunduhan, ekstraksi, dan verifikasi dataset SMDG-19."""

    KAGGLE_DATASET = "deathtrooper/multichannel-glaucoma-benchmark-dataset"

    def __init__(self, target_dir: str = "data/raw/smdg-19"):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def is_dataset_present(self) -> bool:
        """Mengecek apakah direktori sudah memuat file dataset yang valid."""
        # SMDG-19 biasanya memuat folder citra dan file CSV metadata
        csv_files = list(self.target_dir.glob("*.csv"))
        image_files = list(self.target_dir.glob("**/*.png")) + list(self.target_dir.glob("**/*.jpg"))
        if len(csv_files) > 0 and len(image_files) > 10:
            return True
        return False

    def extract_existing_zip(self) -> bool:
        """Mengekstrak jika ditemukan file zip dataset di direktori target atau parent."""
        possible_dirs = [self.target_dir, self.target_dir.parent]
        for p in possible_dirs:
            for zip_path in p.glob("*.zip"):
                if "glaucoma" in zip_path.name.lower() or "smdg" in zip_path.name.lower() or "benchmark" in zip_path.name.lower():
                    print(f"📦 Ditemukan arsip lokal: {zip_path}")
                    print(f"Mengekstrak ke: {self.target_dir} ...")
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(self.target_dir)
                    print("✅ Ekstraksi arsip ZIP selesai.")
                    return True
        return False

    def setup_kaggle_credentials(self, username: str, key: str):
        """Menyiapkan file ~/.kaggle/kaggle.json secara otomatis."""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        creds_file = kaggle_dir / "kaggle.json"
        with open(creds_file, "w") as f:
            json.dump({"username": username, "key": key}, f)
        creds_file.chmod(0o600)
        print(f"✅ Kredensial Kaggle tersimpan di {creds_file}")

    def download_via_kaggle(self) -> bool:
        """Mengunduh menggunakan package kaggle resmi."""
        try:
            import kaggle
            print(f"🚀 Memulai download dataset '{self.KAGGLE_DATASET}' melalui Kaggle API...")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                self.KAGGLE_DATASET,
                path=str(self.target_dir),
                unzip=True,
                quiet=False
            )
            print(f"✅ Berhasil mengunduh dan mengekstrak dataset ke {self.target_dir}")
            return True
        except Exception as e:
            print(f"⚠️ Gagal download via Kaggle API: {e}")
            return False

    def download_via_kagglehub(self) -> bool:
        """Mengunduh menggunakan kagglehub."""
        try:
            import kagglehub
            import shutil
            print(f"🚀 Memulai download dataset '{self.KAGGLE_DATASET}' melalui kagglehub...")
            download_path = kagglehub.dataset_download(self.KAGGLE_DATASET)
            print(f"📦 Dataset diunduh ke cache: {download_path}")
            print(f"Menyalin data ke target direktori: {self.target_dir} ...")
            for item in Path(download_path).iterdir():
                dest = self.target_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            print(f"✅ Data berhasil disalin ke {self.target_dir}")
            return True
        except Exception as e:
            print(f"⚠️ Gagal download via kagglehub: {e}")
            return False

    def generate_synthetic_samples(self, num_samples: int = 24):
        """
        Membuat dataset tiruan (synthetic) berstruktur persis SMDG-19.
        Sangat berguna untuk pengujian cepat end-to-end tanpa menunggu download 3GB+.
        """
        print(f"🧪 Membuat {num_samples} sampel data sintetis SMDG-19 untuk pengujian pipeline...")
        images_dir = self.target_dir / "full-fundus"
        od_dir = self.target_dir / "optic-disc"
        vessel_dir = self.target_dir / "blood-vessel"

        images_dir.mkdir(parents=True, exist_ok=True)
        od_dir.mkdir(parents=True, exist_ok=True)
        vessel_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for i in range(num_samples):
            name = f"sample_{i:04d}"
            # 1. Buat citra fundus sintetis 512x512 RGB
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            # Gambar bola mata retina (lingkaran oranye kemerahan)
            center = (256, 256)
            cv2.circle(img, center, 230, (30, 80, 200), -1)  # BGR
            # Tambahkan tekstur random
            noise = np.random.randint(-15, 15, (512, 512, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # 2. Buat masker Optic Disc sintetis
            od_mask = np.zeros((512, 512), dtype=np.uint8)
            od_center = (210 + (i % 5) * 10, 256 + (i % 3) * 15)
            cv2.ellipse(od_mask, od_center, (45, 55), 10, 0, 360, 255, -1)
            # Pada citra asli, beri warna terang pada area OD
            img[od_mask == 255] = [80, 160, 240]

            # 3. Buat masker pembuluh darah sintetis
            vessel_mask = np.zeros((512, 512), dtype=np.uint8)
            for branch in range(6):
                pt1 = od_center
                angle = (branch * 60 + (i * 7)) * np.pi / 180
                pt2 = (int(od_center[0] + 180 * np.cos(angle)), int(od_center[1] + 180 * np.sin(angle)))
                cv2.line(vessel_mask, pt1, pt2, 255, thickness=np.random.randint(2, 5))
                # Cabang kedua
                pt3 = (int(pt2[0] + 50 * np.cos(angle + 0.5)), int(pt2[1] + 50 * np.sin(angle + 0.5)))
                cv2.line(vessel_mask, pt2, pt3, 255, thickness=2)

            # Terapkan pembuluh darah gelap pada citra fundus
            img[vessel_mask == 255] = [20, 30, 80]

            # Simpan file gambar
            img_path = images_dir / f"{name}.png"
            od_path = od_dir / f"{name}_od.png"
            vessel_path = vessel_dir / f"{name}_vessel.png"

            cv2.imwrite(str(img_path), img)
            cv2.imwrite(str(od_path), od_mask)
            cv2.imwrite(str(vessel_path), vessel_mask)

            # Label: 0 = Normal, 1 = Glaukoma
            label = 1 if i % 3 == 0 else 0
            records.append({
                "image_id": name,
                "fundus_path": str(img_path.relative_to(self.target_dir)),
                "od_mask_path": str(od_path.relative_to(self.target_dir)),
                "vessel_mask_path": str(vessel_path.relative_to(self.target_dir)),
                "glaucoma_label": label,
                "dataset_source": "SYNTHETIC-SMDG"
            })

        # Simpan metadata CSV
        df = pd.DataFrame(records)
        csv_path = self.target_dir / "metadata_smdg19.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Metadata sintetis disimpan di: {csv_path}")
        print(f"Distribusi label: Normal={sum(df.glaucoma_label == 0)}, Glaukoma={sum(df.glaucoma_label == 1)}")


def main():
    parser = argparse.ArgumentParser(description="SMDG-19 Dataset Downloader & Manager")
    parser.add_argument("--target-dir", default="data/raw/smdg-19", help="Direktori penyimpanan dataset")
    parser.add_argument("--kaggle-user", default=None, help="Kaggle username")
    parser.add_argument("--kaggle-key", default=None, help="Kaggle API key")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate sampel sintetis untuk unit testing")
    parser.add_argument("--num-samples", type=int, default=24, help="Jumlah sampel sintetis")
    args = parser.parse_args()

    downloader = SMDGDownloader(target_dir=args.target_dir)

    if args.generate_synthetic:
        downloader.generate_synthetic_samples(num_samples=args.num_samples)
        return

    if downloader.is_dataset_present():
        print(f"✅ Dataset SMDG-19 sudah terdeteksi di {args.target_dir}!")
        return

    print("🔍 Memeriksa apakah ada arsip lokal ZIP...")
    if downloader.extract_existing_zip():
        return

    if args.kaggle_user and args.kaggle_key:
        downloader.setup_kaggle_credentials(args.kaggle_user, args.kaggle_key)

    # Coba download via Kagglehub / Kaggle API
    success = downloader.download_via_kagglehub()
    if not success:
        success = downloader.download_via_kaggle()

    if not success:
        print("\n" + "=" * 65)
        print("ℹ️ PANDUAN PENGUNDUHAN DATASET SMDG-19:")
        print("1. Anda dapat mengunduh manual dari:")
        print("   https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset")
        print(f"   Lalu letakkan file ZIP atau folder hasil ekstraksi di '{args.target_dir}'.")
        print("\n2. Atau jalankan dengan memasukkan kredensial Kaggle Anda:")
        print("   python3 -m src.dataset.downloader --kaggle-user <USERNAME> --kaggle-key <KEY>")
        print("\n3. Atau buat dataset sintetis untuk menguji pipeline sekarang:")
        print("   python3 -m src.dataset.downloader --generate-synthetic")
        print("=" * 65)


if __name__ == "__main__":
    main()
