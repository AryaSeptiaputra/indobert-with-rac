# 4.4 Hasil Pengujian pada Data Uji (Test Set)

Konfigurasi terbaik dari masing-masing skenario (§4.3), yang dipilih murni berdasarkan
`val_f1_macro`, dievaluasi satu kali pada *test set* (1.405 baris, 1.149 non-judi / 256 judi)
untuk mengukur generalisasi model pada data yang belum pernah dilihat sama sekali selama
proses tuning — sesuai prinsip *test set* hanya dibuka sekali di akhir (§4.1).

## 4.4.1 Hasil Klasifikasi pada Test Set

| Model | Konfigurasi final | test F1-macro | test Accuracy | test F1 judi | test Precision judi | test Recall judi |
|---|---|---|---|---|---|---|
| **RM-a** | `lr`=2e-5, `epochs`=5, `batch`=32 | **0,9607** | 0,9765 | 0,9357 | 0,9339 | 0,9375 |
| **RM-b** | MLP/1024, `epochs`=10, `lr`=1e-3 | 0,9486 | 0,9694 | 0,9159 | 0,9176 | 0,9141 |
| **RM-c** | head RM-b + α=0,2, k=5 | 0,9497 | 0,9701 | 0,9176 | 0,9213 | 0,9141 |

Tabel 4.9 Hasil klasifikasi pada *test set* untuk ketiga skenario (`results/vast/metrics/final_comparison.csv`)

Visualisasi *confusion matrix* dan kurva *precision-recall* kelas judi untuk masing-masing
model tersedia pada `results/vast/figures/final_rm{a,b,c}_confusion.png` dan
`final_rm{a,b,c}_pr.png`.

## 4.4.2 Konsistensi Peringkat Validation vs Test

| Model | val F1-macro | test F1-macro | Selisih (val → test) |
|---|---|---|---|
| RM-a | 0,9822 | 0,9607 | −2,15pp |
| RM-b | 0,9653 | 0,9486 | −1,67pp |
| RM-c | 0,9699 | 0,9497 | −2,02pp |

Tabel 4.10 Selisih performa validation ke test per skenario

Peringkat ketiga skenario pada *test set* (RM-a > RM-c > RM-b) **konsisten** dengan peringkat
pada *validation set* yang menjadi dasar pemilihan hyperparameter — tidak terjadi pertukaran
urutan (*rank inversion*) antara val dan test. Konsistensi ini menjadi indikasi bahwa
keputusan tuning yang diambil murni dari `val_f1_macro` (§4.3) tidak mengalami *overfitting*
berarti terhadap *validation set*, meski seluruh skenario mengalami penurunan performa yang
wajar (1,67–2,15pp) saat berpindah ke data yang benar-benar belum pernah dilihat.

RM-a tetap unggul di *test set* sebagaimana di *validation set*, namun **gap-nya terhadap
RM-b dan RM-c menyempit** dibanding gap pada *val* (dari ~1,7–1,9pp di val menjadi ~1,1–1,2pp
di test) — mengindikasikan bahwa dua strategi ringan tersebut menggeneralisasi setidaknya
sebaik, jika bukan sedikit lebih baik secara relatif, dibanding *full fine-tuning* pada data
uji ini. Implikasi lebih lanjut dari selisih performa ini terhadap klaim *trade-off*
efisiensi dibahas pada §4.7.
