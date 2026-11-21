# Rangkuman Pengujian Independen (SMDG-19 Test Set)

## 1. Kinerja Segmentasi Tahap 1 (Optic Disc & Pembuluh Darah)
| Model Variant | Target Organ | Dice (F1) | IoU (Jaccard) | Recall | Specificity | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla DUNet | Optic Disc | **0.9495** | 0.9051 | 0.9455 | 0.9990 | 0.9979 |
| Hybrid DUNet | Optic Disc | **0.9375** | 0.8843 | 0.9394 | 0.9986 | 0.9973 |
| Hybrid DUNet | Optic Cup | **0.8362** | 0.7265 | 0.8428 | 0.9992 | 0.9983 |
| Vanilla DUNet | Blood Vessel | **0.8945** | 0.8113 | 0.8848 | 0.9925 | 0.9849 |
| Hybrid DUNet | Blood Vessel | **0.8565** | 0.7511 | 0.8596 | 0.9884 | 0.9792 |

## 2. Kinerja Klasifikasi Tahap 2 & Studi Ablasi Loss
| Konfigurasi Detektor | Loss Function | AUC-ROC | Accuracy | Precision | Recall | F1-Score | Specificity |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla_Scenario1_CE | Multi-GlaucNet | **0.9548** | 0.8975 | 0.8925 | 0.8352 | 0.8629 | 0.9367 |
| Vanilla_Scenario2_StdFocal | Multi-GlaucNet | **0.9460** | 0.8937 | 0.8788 | 0.8408 | 0.8594 | 0.9270 |
| Vanilla_Scenario3_ExtremeFocal | Multi-GlaucNet | **0.9539** | 0.8824 | 0.8329 | 0.8701 | 0.8511 | 0.8901 |
| Hybrid_Scenario1_CE | Multi-GlaucNet | **0.9527** | 0.8931 | 0.9137 | 0.7989 | 0.8525 | 0.9525 |
| Hybrid_Scenario2_Proposed | Multi-GlaucNet | **0.9441** | 0.8942 | 0.8757 | 0.8464 | 0.8608 | 0.9244 |
| Hybrid_Scenario3_ExtremeFocal | Multi-GlaucNet | **0.9494** | 0.8851 | 0.8376 | 0.8715 | 0.8542 | 0.8936 |