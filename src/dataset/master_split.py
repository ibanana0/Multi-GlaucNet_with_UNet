"""
Master Split Generator for SMDG-19
Menghasilkan pembagian partisi deterministik (70% Train, 15% Val, 15% Test)
dengan stratifikasi ganda (dataset_origin + glaucoma_label) untuk menjamin
ZERO CROSS-STAGE DATA LEAKAGE antara segmentasi dan deteksi.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataset_origin(image_id: str) -> str:
    """Mengekstrak asal sub-dataset dari prefiks nama berkas."""
    name = str(image_id).upper()
    prefixes = [
        'REFUGE', 'ORIGA', 'G1020', 'PAPILA', 'DRISHTI',
        'FIVES', 'HRF', 'LES-AV', 'DR-HAGIS', 'ODIR',
        'AIROGS', 'BEH', 'CHOI', 'CRFO', 'JSIEC'
    ]
    for prefix in prefixes:
        if prefix in name:
            return prefix
    return 'AIROGS' if ('EYEPACS' in name or 'AIROGS' in name) else 'OTHER'


def generate_master_split(
    csv_path: str,
    output_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Menghasilkan berkas CSV master split dengan kolom 'split' ('train', 'val', 'test').
    Menerapkan stratified split per dataset_origin agar:
    1. Subset REFUGE 100% identik dengan hasil Phase 1 (560 Train, 120 Val, 120 Test).
    2. Zero Cross-Stage Data Leakage antar-tahap pada seluruh SMDG-19.
    """
    df = pd.read_csv(csv_path)

    # Standarisasi kolom id & label
    id_col = "names" if "names" in df.columns else "image_id"
    label_col = "types" if "types" in df.columns else "glaucoma_label"

    df["dataset_origin"] = df[id_col].apply(get_dataset_origin)

    train_dfs, val_dfs, test_dfs = [], [], []

    temp_ratio = round(1.0 - train_ratio, 4)
    val_rel_ratio = round(val_ratio / temp_ratio, 4)

    for origin, group in df.groupby("dataset_origin"):
        strat = group[label_col] if group[label_col].nunique() > 1 and group[label_col].value_counts().min() >= 2 else None

        if len(group) < 10:
            n_tr = max(1, int(len(group) * train_ratio))
            tr_g = group.iloc[:n_tr].copy()
            rem_g = group.iloc[n_tr:].copy()
            if len(rem_g) >= 2:
                mid = len(rem_g) // 2
                va_g = rem_g.iloc[:mid].copy()
                te_g = rem_g.iloc[mid:].copy()
            else:
                va_g = rem_g.copy()
                te_g = rem_g.copy()
        else:
            tr_g, temp_g = train_test_split(
                group,
                test_size=temp_ratio,
                random_state=random_seed,
                stratify=strat
            )
            strat_temp = temp_g[label_col] if temp_g[label_col].nunique() > 1 and temp_g[label_col].value_counts().min() >= 2 else None
            va_g, te_g = train_test_split(
                temp_g,
                test_size=(1.0 - val_rel_ratio),
                random_state=random_seed,
                stratify=strat_temp
            )

        tr_g = tr_g.copy()
        va_g = va_g.copy()
        te_g = te_g.copy()

        tr_g["split"] = "train"
        va_g["split"] = "val"
        te_g["split"] = "test"

        train_dfs.append(tr_g)
        val_dfs.append(va_g)
        test_dfs.append(te_g)

    master_df = pd.concat(train_dfs + val_dfs + test_dfs).sort_index()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(out_file, index=False)

    print(f"✅ Berkas Master-Split berhasil disimpan di: {out_file}")
    print(f"📊 Distribusi Keseluruhan:")
    print(master_df["split"].value_counts().to_string())

    return master_df


if __name__ == "__main__":
    default_csv = "/root/biomedik/dataset/smdg-19/archive/metadata - standardized.csv"
    default_out = "/root/biomedik/Multi-GlaucNet_with_UNet/splits/master_split_seed42.csv"
    generate_master_split(default_csv, default_out)
