"""
Retinal Preprocessing Module according to MultiGlaucNet Phase 1:
- PRE-01: Uniform Resizing (512x512)
- PRE-02: Retinal ROI Cropping (removing non-retina black borders)
- PRE-03: CLAHE (Contrast-Limited Adaptive Histogram Equalization)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class RetinalPreprocessor:
    """Modul pre-processing citra fundus retina dan masker ground-truth."""

    def __init__(self, target_size: Tuple[int, int] = (512, 512), clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.target_size = target_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def crop_roi(self, image: np.ndarray, masks: Optional[Dict[str, np.ndarray]] = None, tolerance: int = 10) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        PRE-02: Memotong area background hitam non-retina di sekitar bola mata.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        retina_mask = gray > tolerance

        if not np.any(retina_mask):
            return image, masks

        rows = np.any(retina_mask, axis=1)
        cols = np.any(retina_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Berikan margin kecil (padding) jika memungkinkan
        h, w = gray.shape
        pad = 5
        rmin = max(0, rmin - pad)
        rmax = min(h, rmax + pad + 1)
        cmin = max(0, cmin - pad)
        cmax = min(w, cmax + pad + 1)

        cropped_img = image[rmin:rmax, cmin:cmax]

        cropped_masks = None
        if masks is not None:
            cropped_masks = {}
            for k, m in masks.items():
                if m is not None:
                    # Pastikan dimensi masker selaras dengan citra sebelum di-crop
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    cropped_masks[k] = m[rmin:rmax, cmin:cmax]
                else:
                    cropped_masks[k] = None

        return cropped_img, cropped_masks

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        PRE-03: Menerapkan CLAHE pada kanal L (Luminance) dalam ruang warna LAB,
        sehingga meningkatkan kontras pembuluh darah dan batas Optic Disc tanpa merusak warna alami.
        """
        if len(image.shape) == 2:
            return self.clahe.apply(image)

        # Ubah RGB ke LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        enhanced_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return enhanced_rgb

    def resize(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """
        PRE-01: Meresize citra ke resolusi target uniform (default 512x512).
        """
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(image, self.target_size, interpolation=interp)

    def process(self, image: np.ndarray, od_mask: Optional[np.ndarray] = None, vessel_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Menjalankan seluruh pipeline pre-processing (ROI Crop -> CLAHE -> Resize 512x512).
        """
        masks = {}
        if od_mask is not None:
            masks["od"] = od_mask
        if vessel_mask is not None:
            masks["vessel"] = vessel_mask

        # 1. ROI Crop
        cropped_img, cropped_masks = self.crop_roi(image, masks=masks if len(masks) > 0 else None)

        # 2. CLAHE pada citra fundus
        enhanced_img = self.apply_clahe(cropped_img)

        # 3. Resize ke 512x512
        final_img = self.resize(enhanced_img, is_mask=False)

        result = {"image": final_img}

        if cropped_masks:
            if "od" in cropped_masks and cropped_masks["od"] is not None:
                resized_od = self.resize(cropped_masks["od"], is_mask=True)
                # Binarisasi masker ground truth jika rentang 0-255
                result["od_mask"] = (resized_od > 127).astype(np.float32)
            if "vessel" in cropped_masks and cropped_masks["vessel"] is not None:
                resized_vessel = self.resize(cropped_masks["vessel"], is_mask=True)
                result["vessel_mask"] = (resized_vessel > 127).astype(np.float32)

        return result
