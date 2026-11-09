# Rangkuman Pengujian Independen (SMDG-19 Test Set)

## 1. Kinerja Segmentasi Tahap 1 (Optic Disc & Pembuluh Darah)
| Model Variant | Target Organ | Dice (F1) | IoU (Jaccard) | Recall | Specificity | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla DUNet | Optic Disc | **0.9520** | 0.9094 | 0.9481 | 0.9991 | 0.9980 |
| Hybrid DUNet | Optic Disc | **0.9501** | 0.9061 | 0.9490 | 0.9990 | 0.9979 |
| Vanilla DUNet | Blood Vessel | **0.8881** | 0.8000 | 0.8958 | 0.9900 | 0.9830 |
| Hybrid DUNet | Blood Vessel | **0.8449** | 0.7326 | 0.8746 | 0.9839 | 0.9757 |

## 2. Kinerja Klasifikasi Tahap 2 & Studi Ablasi Loss
| Konfigurasi Detektor | Loss Function | AUC-ROC | Accuracy | Precision | Recall | F1-Score | Specificity |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla_Scenario1_CE | Multi-GlaucNet | **0.9810** | 0.9156 | 0.8463 | **0.9552** | **0.8975** | 0.8906 |
| Vanilla_Scenario2_StdFocal | Multi-GlaucNet | 0.9742 | **0.9183** | **0.8821** | 0.9105 | 0.8961 | **0.9232** |
| Vanilla_Scenario3_ExtremeFocal | Multi-GlaucNet | 0.9801 | 0.9096 | 0.8391 | 0.9483 | 0.8903 | 0.8853 |
| Hybrid_Scenario2_Proposed | Multi-GlaucNet | **0.9773** | 0.9118 | 0.8494 | 0.9385 | 0.8917 | 0.8950 |