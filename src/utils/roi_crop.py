"""
Optic Disc Square Tight ROI Cropping Utility for MultiGlaucNet (Branch fix3).
Eliminates 95% black-canvas background dilution in Global Average Pooling while
preserving strict 1:1 aspect ratio for accurate vertical Cup-to-Disc Ratio (vCDR) analysis.
"""

import torch
import torch.nn.functional as F


def crop_od_roi(
    I_orig: torch.Tensor,
    M_od: torch.Tensor,
    margin_ratio: float = 0.20,
    mask_background: bool = True,
    min_box_size: int = 32,
) -> torch.Tensor:
    """
    Memotong Optic Disc secara ketat dalam kotak persegi (rasio aspek 1:1) dari kanvas penuh,
    lalu me-resize kembali ke resolusi input (H, W) agar kompatibel dengan ThreeBranchResNet50.

    Args:
        I_orig: Tensor citra fundus [B, C, H, W]
        M_od: Tensor masker biner Optic Disc [B, 1, H, W]
        margin_ratio: Proporsi margin tambahan di setiap sisi bounding box (default 0.20 = 20% per sisi)
        mask_background: Jika True, terapkan masking (I_orig * M_od) sebelum cropping
                         sehingga sudut di luar lingkaran OD tetap bernilai 0.
        min_box_size: Ukuran minimum sisi kotak crop dalam piksel.

    Returns:
        Tensor I_od_cropped [B, C, H, W] ter-resize ke resolusi asli (H, W).
    """
    B, _, H, W = I_orig.shape
    source = I_orig * M_od if mask_background else I_orig
    cropped_list = []

    for b in range(B):
        mask_2d = M_od[b, 0] > 0.5
        ys, xs = torch.where(mask_2d)

        if ys.numel() == 0:
            # Fallback aman jika segmentasi OD gagal/kosong: center crop 50% kanvas
            side = min(H, W) // 2
            y1 = (H - side) // 2
            x1 = (W - side) // 2
            patch = I_orig[b : b + 1, :, y1 : y1 + side, x1 : x1 + side]
        else:
            y_min = int(ys.min().item())
            y_max = int(ys.max().item())
            x_min = int(xs.min().item())
            x_max = int(xs.max().item())

            box_h = y_max - y_min + 1
            box_w = x_max - x_min + 1
            base_side = max(box_h, box_w, min_box_size)

            # Tambahkan margin simetris di kedua sisi, lalu kunci sebagai persegi 1:1
            padded_side = int(round(base_side * (1.0 + 2.0 * margin_ratio)))
            padded_side = max(min_box_size, min(padded_side, min(H, W)))

            cy = 0.5 * (y_min + y_max)
            cx = 0.5 * (x_min + x_max)

            half = padded_side // 2
            y1 = int(round(cy)) - half
            x1 = int(round(cx)) - half

            # Geser kotak agar tetap di dalam batas citra tanpa mengubah ukuran sisi (menjaga rasio 1:1)
            y1 = max(0, min(y1, H - padded_side))
            x1 = max(0, min(x1, W - padded_side))

            patch = source[b : b + 1, :, y1 : y1 + padded_side, x1 : x1 + padded_side]

        resized = F.interpolate(patch, size=(H, W), mode="bilinear", align_corners=False)
        cropped_list.append(resized)

    return torch.cat(cropped_list, dim=0)
