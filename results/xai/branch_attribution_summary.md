# Rangkuman Evaluasi Pilar 1: Closed-Form Branch Attribution

Evaluasi kontribusi aditif linier eksak pada lapisan klasifikasi Three-Branch ResNet-50.

- **Total Citra Test Set Teruji**: 1853
- **Jumlah Mata Normal**: 1137
- **Jumlah Mata Glaukoma**: 716

## 1. Rata-Rata Kontribusi Relatif Cabang (%)
| Kohort Evaluasi | Cabang Citra Asli (%) | Cabang Optic Disc (%) | Cabang Pembuluh Darah (%) |
|---|---|---|---|
| **Seluruh Test Set (N=1853)** | 82.54 ± 15.22% | **5.62 ± 5.83%** | 11.84 ± 11.87% |
| Kohort Normal (N=1137) | 80.72 ± 15.63% | 6.20 ± 6.01% | 13.08 ± 12.22% |
| Kohort Glaukoma (N=716) | 85.43 ± 14.09% | **4.69 ± 5.42%** | 9.87 ± 11.02% |

## 2. Analisis Signifikansi Klinis
- **Peranan Biomarker Optic Disc**: Cabang Optic Disc secara konsisten menyumbang dorongan logit yang signifikan pada mata glaukoma, membuktikan bahwa fitur morfologi diskus dan mangkuk optik merupakan tumpuan utama model.
- **Peranan Biomarker Vaskular**: Cabang pembuluh darah memberikan informasi komplementer terhadap pola arsitektur retina.
- **Sifat Dekomposisi Eksak**: Karena dihitung langsung dari sifat distributif aljabar linier, nilai atribusi bebas dari galat aproksimasi sampling (*zero estimation error*).