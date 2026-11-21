"""
Explainable AI (XAI) Module for MultiGlaucNet:
Implements 3 Pillars of Interpretability:
1. Closed-Form Branch Attribution (Exact Additive Shapley Decomposition for Three-Branch Late Fusion).
2. Multi-Branch Grad-CAM (Spatial Attribution across Global, Optic Disc, and Vascular Branches).
3. Clinical Sanity Check & Correlation Audit (Verification of Physiological Receptors & Anti-Shortcut).
"""

import sys
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClosedFormBranchAttribution:
    """
    Pilar 1: Menghitung dekomposisi aditif eksak (Closed-Form Additive Attribution)
    dari Three-Branch ResNet-50 tanpa aproksimasi sampling (Lundberg & Lee, NeurIPS 2017).

    Untuk fusi rata-rata elemen f_fused = (f_orig + f_od + f_vessel) / 3:
        z_c = (1/3 * w_c^T * f_orig) + (1/3 * w_c^T * f_od) + (1/3 * w_c^T * f_vessel) + b_c
    """

    def __init__(self, detector: nn.Module):
        self.detector = detector
        self.detector.eval()

        # Ambil lapisan linier terakhir
        if hasattr(self.detector, "classifier"):
            if isinstance(self.detector.classifier, nn.Sequential):
                # Cari layer Linear di dalam Sequential
                linear_layers = [m for m in self.detector.classifier if isinstance(m, nn.Linear)]
                if not linear_layers:
                    raise ValueError("Tidak ditemukan nn.Linear di dalam detector.classifier.")
                self.fc = linear_layers[-1]
            elif isinstance(self.detector.classifier, nn.Linear):
                self.fc = self.detector.classifier
            else:
                raise TypeError(f"Tipe detector.classifier tidak dikenali: {type(self.detector.classifier)}")
        elif hasattr(self.detector, "fc") and isinstance(self.detector.fc, nn.Linear):
            self.fc = self.detector.fc
        else:
            raise AttributeError("Model tidak memiliki atribut classifier atau fc yang bertipe nn.Linear.")

        self.weight = self.fc.weight.detach()  # [num_classes, in_features]
        self.bias = self.fc.bias.detach() if self.fc.bias is not None else None
        self.fusion_method = getattr(self.detector, "fusion_method", "average").lower()
        self.feature_dim = getattr(self.detector, "feature_dim", 2048)

    @torch.no_grad()
    def compute_attribution(
        self,
        I_orig: torch.Tensor,
        I_od: torch.Tensor,
        I_vessel: torch.Tensor,
        target_class: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Menghitung skor kontribusi logit parsial dan persentase kontribusi cabang untuk satu batch.

        Args:
            I_orig: Tensor citra asli [B, 3, H, W]
            I_od: Tensor citra masking Optic Disc [B, 3, H, W]
            I_vessel: Tensor citra masking Pembuluh Darah [B, 3, H, W]
            target_class: Indeks kelas target (default: 1 untuk Glaukoma)

        Returns:
            Dict berisi:
                - S_orig: Kontribusi logit cabang original [B]
                - S_od: Kontribusi logit cabang Optic Disc [B]
                - S_vessel: Kontribusi logit cabang Pembuluh Darah [B]
                - bias: Bias kelas target [B]
                - logit_sum: Total logit rekonstruksi [B]
                - pct_orig: Persentase kontribusi absolut cabang original [B]
                - pct_od: Persentase kontribusi absolut cabang Optic Disc [B]
                - pct_vessel: Persentase kontribusi absolut cabang Pembuluh Darah [B]
        """
        # Ekstraksi fitur 2048-d dari masing-masing cabang
        f_orig, f_od, f_vessel = self.detector.extract_features(I_orig, I_od, I_vessel)
        b_size = f_orig.size(0)
        device = f_orig.device

        w = self.weight.to(device)[target_class]  # [in_features]
        b_val = self.bias.to(device)[target_class] if self.bias is not None else torch.tensor(0.0, device=device)
        bias_vec = b_val.expand(b_size)

        if self.fusion_method == "concat":
            # Concat: w dipecah menjadi 3 segmen dimensi
            dim = self.feature_dim
            w_orig = w[:dim]
            w_od = w[dim:2*dim]
            w_vessel = w[2*dim:]

            S_orig = torch.matmul(f_orig, w_orig)
            S_od = torch.matmul(f_od, w_od)
            S_vessel = torch.matmul(f_vessel, w_vessel)
        else:
            # Average fusion: pembobotan merata 1/3
            coeff = 1.0 / 3.0
            S_orig = coeff * torch.matmul(f_orig, w)
            S_od = coeff * torch.matmul(f_od, w)
            S_vessel = coeff * torch.matmul(f_vessel, w)

        logit_sum = S_orig + S_od + S_vessel + bias_vec

        # Hitung rasio kontribusi absolut (mengukur kekuatan dorongan sinyal cabang)
        abs_orig = torch.abs(S_orig)
        abs_od = torch.abs(S_od)
        abs_vessel = torch.abs(S_vessel)
        total_abs = abs_orig + abs_od + abs_vessel + 1e-8

        pct_orig = (abs_orig / total_abs) * 100.0
        pct_od = (abs_od / total_abs) * 100.0
        pct_vessel = (abs_vessel / total_abs) * 100.0

        return {
            "S_orig": S_orig,
            "S_od": S_od,
            "S_vessel": S_vessel,
            "bias": bias_vec,
            "logit_sum": logit_sum,
            "pct_orig": pct_orig,
            "pct_od": pct_od,
            "pct_vessel": pct_vessel
        }


class MultiBranchGradCAM:
    """
    Pilar 2: Multi-Branch Grad-CAM untuk visualisasi atribusi spasial pada ketiga cabang
    (Citra Asli, Masker OD, dan Masker Pembuluh Darah) secara simultan.
    """

    def __init__(self, detector: nn.Module):
        self.detector = detector
        self.detector.eval()

        # Hooks storage
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.hook_handles = []

        self._register_hooks()

    def _register_hooks(self):
        """Memasang forward hook dan tensor backward hook pada layer4 di ketiga cabang."""
        target_branches = {
            "orig": self.detector.branch_orig.layer4,
            "od": self.detector.branch_od.layer4,
            "vessel": self.detector.branch_vessel.layer4
        }

        for name, layer in target_branches.items():
            def make_fwd_hook(branch_name):
                def hook(module, inp, out):
                    self.activations[branch_name] = out
                    if out.requires_grad:
                        out.register_hook(lambda grad: self.gradients.__setitem__(branch_name, grad))
                return hook

            h = layer.register_forward_hook(make_fwd_hook(name))
            self.hook_handles.append(h)

    def remove_hooks(self):
        """Melepaskan seluruh hook untuk mencegah kebocoran memori."""
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def generate_cam(
        self,
        I_orig: torch.Tensor,
        I_od: torch.Tensor,
        I_vessel: torch.Tensor,
        target_class: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Menghasilkan heatmap Grad-CAM untuk 1 sampel citra (batch_size = 1).

        Returns:
            Dict: {"orig": cam_orig, "od": cam_od, "vessel": cam_vessel}
            Setiap cam adalah matriks numpy 2D [H, W] ternormalisasi ke skala [0.0, 1.0].
        """
        assert I_orig.size(0) == 1, "Grad-CAM per-sampel membutuhkan batch_size = 1."
        H, W = I_orig.shape[2], I_orig.shape[3]

        self.detector.zero_grad()
        self.activations.clear()
        self.gradients.clear()

        # Memastikan aktivasi melacak gradien
        with torch.enable_grad():
            f_orig, f_od, f_vessel = self.detector.extract_features(I_orig, I_od, I_vessel)

            if getattr(self.detector, "fusion_method", "average") == "concat":
                f_fused = torch.cat([f_orig, f_od, f_vessel], dim=1)
            else:
                f_fused = (f_orig + f_od + f_vessel) / 3.0

            logits = self.detector.classifier(f_fused)
            score = logits[0, target_class]
            score.backward(retain_graph=False)

        cams = {}
        for name in ["orig", "od", "vessel"]:
            act = self.activations.get(name)
            grad = self.gradients.get(name)

            if act is None or grad is None:
                # Fallback nol jika gradien tidak tersadap
                cams[name] = np.zeros((H, W), dtype=np.float32)
                continue

            # Global average pooling gradien pada dimensi spasial (H', W')
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, 2048, 1, 1]
            cam = torch.relu(torch.sum(weights * act, dim=1, keepdim=True))  # [1, 1, H', W']

            # Interpolasi bilineer ke ukuran citra asli [H, W]
            cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()

            # Normalisasi Min-Max ke rentang [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            else:
                cam = np.zeros_like(cam)

            cams[name] = cam.astype(np.float32)

        return cams


class ClinicalAuditor:
    """
    Pilar 3: Clinical Sanity Check & Correlation Audit.
    Memverifikasi keselarasan atensi model dengan biomarker oftalmologi dan menguji integritas anti-shortcut.
    """

    @staticmethod
    def audit_retinal_boundary_energy(cam: np.ndarray, tolerance_margin: float = 0.05) -> Dict[str, float]:
        """
        Mengaudit apakah atensi model terfokus di dalam lingkaran retina atau terdistraksi
        oleh batas hitam sirkular kamera fundus (shortcut learning).

        Args:
            cam: Heatmap Grad-CAM 2D [H, W] bernilai [0, 1]
            tolerance_margin: Margin toleransi tepi lingkaran

        Returns:
            Dict berisi:
                - inside_energy_ratio: Persentase energi atensi di dalam area retina (target > 85%)
                - boundary_energy_ratio: Persentase energi di area bingkai luar
                - passes_audit: Boolean apakah lolos verifikasi anti-shortcut
        """
        H, W = cam.shape
        cy, cx = H // 2, W // 2
        radius = min(cy, cx) * (1.0 - tolerance_margin)

        y, x = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        retina_mask = dist_from_center <= radius

        total_energy = np.sum(cam) + 1e-8
        inside_energy = np.sum(cam[retina_mask])
        boundary_energy = np.sum(cam[~retina_mask])

        inside_ratio = float((inside_energy / total_energy) * 100.0)
        boundary_ratio = float((boundary_energy / total_energy) * 100.0)

        return {
            "inside_energy_ratio": inside_ratio,
            "boundary_energy_ratio": boundary_ratio,
            "passes_audit": bool(inside_ratio >= 85.0)
        }

    @staticmethod
    def compute_clinical_correlations(
        branch_scores: np.ndarray,
        clinical_markers: np.ndarray
    ) -> Dict[str, float]:
        """
        Menghitung korelasi Pearson dan Spearman antara kontribusi cabang (misal S_OD)
        dengan penanda klinis (misal Vertical Cup-to-Disc Ratio / vCDR).
        """
        from scipy.stats import pearsonr, spearmanr

        # Filter NaN jika ada
        valid_mask = np.isfinite(branch_scores) & np.isfinite(clinical_markers)
        if np.sum(valid_mask) < 3:
            return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0, "spearman_p": 1.0}

        x = branch_scores[valid_mask]
        y = clinical_markers[valid_mask]

        p_r, p_p = pearsonr(x, y)
        s_rho, s_p = spearmanr(x, y)

        return {
            "pearson_r": float(p_r),
            "pearson_p": float(p_p),
            "spearman_rho": float(s_rho),
            "spearman_p": float(s_p)
        }
