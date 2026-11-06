"""
PyTorch Dataset and DataLoader for SMDG-19 retinal images.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .preprocessor import RetinalPreprocessor


class SMDGDataset(Dataset):
    """Dataset class untuk memuat citra fundus, masker OD, masker vaskular, dan label glaukoma."""

    def __init__(self, df: pd.DataFrame, root_dir: str, preprocessor: Optional[RetinalPreprocessor] = None, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.preprocessor = preprocessor if preprocessor is not None else RetinalPreprocessor()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return self.root_dir / p

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # 1. Muat Citra Fundus Asli (RGB)
        img_path = self._resolve_path(row["fundus_path"])
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Tidak dapat membaca file citra: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Muat Masker OD jika tersedia
        od_mask = None
        if "od_mask_path" in row and pd.notna(row["od_mask_path"]):
            od_p = self._resolve_path(row["od_mask_path"])
            if od_p.exists():
                od_mask = cv2.imread(str(od_p), cv2.IMREAD_GRAYSCALE)

        # 3. Muat Masker Pembuluh Darah jika tersedia
        vessel_mask = None
        if "vessel_mask_path" in row and pd.notna(row["vessel_mask_path"]):
            vessel_p = self._resolve_path(row["vessel_mask_path"])
            if vessel_p.exists():
                vessel_mask = cv2.imread(str(vessel_p), cv2.IMREAD_GRAYSCALE)

        # 4. Terapkan Preprocessing Pipeline (ROI Crop + CLAHE + Resize 512x512)
        processed = self.preprocessor.process(img_rgb, od_mask=od_mask, vessel_mask=vessel_mask)

        final_img = processed["image"].astype(np.float32) / 255.0  # Normalisasi [0, 1]
        # Transpose HWC -> CHW untuk PyTorch
        tensor_img = torch.from_numpy(final_img).permute(2, 0, 1)

        # Tensor Masker
        if "od_mask" in processed:
            tensor_od = torch.from_numpy(processed["od_mask"]).unsqueeze(0)  # [1, 512, 512]
        else:
            tensor_od = torch.zeros((1, 512, 512), dtype=torch.float32)

        if "vessel_mask" in processed:
            tensor_vessel = torch.from_numpy(processed["vessel_mask"]).unsqueeze(0)  # [1, 512, 512]
        else:
            tensor_vessel = torch.zeros((1, 512, 512), dtype=torch.float32)

        # Label Glaukoma (0 = Normal, 1 = Glaukoma)
        raw_label = int(row.get("glaucoma_label", 0))
        # Jika ada label -1 (suspect), di-mapping ke 1 atau 0 sesuai konvensi biner (default: >0 dianggap glaukoma/suspect)
        binary_label = 1 if raw_label == 1 else 0

        return {
            "image": tensor_img,
            "od_mask": tensor_od,
            "vessel_mask": tensor_vessel,
            "label": torch.tensor(binary_label, dtype=torch.long),
            "image_id": row.get("image_id", str(idx))
        }


def get_dataloaders(csv_path: str, root_dir: str, batch_size: int = 4, train_ratio: float = 0.70, val_ratio: float = 0.15, random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    Membuat DataLoader untuk Train, Val, dan Test set dengan Stratified Split.
    """
    df = pd.read_csv(csv_path)

    # Filter data jika ada label NaN
    if "glaucoma_label" in df.columns:
        df = df[df["glaucoma_label"].isin([0, 1])].copy()

    # Stratified Split: Train vs Temp (Val + Test)
    temp_ratio = 1.0 - train_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=df["glaucoma_label"] if "glaucoma_label" in df.columns else None
    )

    # Split Temp menjadi Val dan Test
    val_rel_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_rel_ratio),
        random_state=random_seed,
        stratify=temp_df["glaucoma_label"] if "glaucoma_label" in temp_df.columns else None
    )

    train_ds = SMDGDataset(train_df, root_dir=root_dir)
    val_ds = SMDGDataset(val_df, root_dir=root_dir)
    test_ds = SMDGDataset(test_df, root_dir=root_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"📊 Distribusi Pembagian Dataset:")
    print(f"  - Training Set  : {len(train_ds)} sampel")
    print(f"  - Validation Set: {len(val_ds)} sampel")
    print(f"  - Test Set      : {len(test_ds)} sampel")

    return train_loader, val_loader, test_loader, df
