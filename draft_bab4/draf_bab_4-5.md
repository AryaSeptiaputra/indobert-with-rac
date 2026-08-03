# 4.5 Analisis Efisiensi Komputasi

Empat metrik efisiensi diukur untuk masing-masing skenario pada konfigurasi final:
jumlah *trainable parameters*, waktu latih total, *peak* memori GPU, dan *latency*
inferensi per sampel. Seluruh angka *latency* dan memori diukur dalam **satu sesi GPU
yang sama** (RTX 3090, lihat §4.1) agar dapat dibandingkan secara apel-ke-apel.

## 4.5.1 Perbandingan Efisiensi

| Model | Trainable params | Reduksi params vs RM-a | Waktu latih | Reduksi waktu vs RM-a | Latency inferensi | Peak memory inferensi |
|---|---|---|---|---|---|---|
| RM-a | 109.485.314 | — | 81,1 s | — | 9,40 ms | 1.473,6 MB |
| RM-b | 789.506 | **99,28%** | 11,65 s | **85,64%** | 9,08 ms | 1.273,9 MB |
| RM-c | 0 | **100%** | 0,0 s | **100%** | 10,96 ms | 1.273,9 MB |

Tabel 4.11 Perbandingan efisiensi komputasi ketiga skenario (`results/vast/metrics/final_comparison.csv`, `inference_benchmark.csv`)

## 4.5.2 Parameter dan Waktu Latih: Reduksi Drastis

RM-b membekukan seluruh 109,48 juta parameter encoder dan hanya melatih *head* MLP
(789.506 parameter, 0,72% dari total RM-a) — menghasilkan reduksi *trainable parameters*
99,28% dan reduksi waktu latih 85,64%. RM-c, yang tidak melibatkan pelatihan sama sekali
(murni fusi probabilitas pada saat inferensi, §4.3.3), secara definisi mencatatkan reduksi
100% pada kedua metrik tersebut relatif terhadap RM-a.

## 4.5.3 Latency Inferensi: Temuan Kunci yang Perlu Digarisbawahi

Berbeda dari pola reduksi drastis pada parameter dan waktu latih, **latency inferensi
RM-b dan RM-c tidak menunjukkan perbaikan berarti dibanding RM-a** — RM-b bahkan hanya
0,32 ms lebih cepat (9,08 ms vs 9,40 ms), sementara RM-c justru **0,56 ms lebih lambat**
dari RM-a (menambahkan biaya pencarian indeks FAISS di atas *forward pass* encoder).

Penyebabnya adalah **arsitektur encoder yang dipakai RM-b/RM-c sama persis dengan RM-a**
(`indobert-base-p2`, 110 juta parameter) — membekukan bobot encoder (`requires_grad=False`)
menghentikan komputasi gradien dan pembaruan bobot saat *training*, tetapi **tidak
mengurangi komputasi *forward pass* saat inferensi**, karena *forward pass* tetap harus
melewati seluruh 12 *layer transformer* yang sama untuk menghasilkan representasi yang akan
diklasifikasikan. Dengan kata lain, efisiensi RM-b/RM-c yang teramati pada §4.5.2 murni
berasal dari sisi *pelatihan* (bobot yang diperbarui dan dihitung gradiennya jauh lebih
sedikit), bukan dari sisi *komputasi inferensi* — sehingga tidak dapat diasumsikan bahwa
reduksi parameter secara otomatis berarti prediksi yang lebih cepat.

Temuan ini menjadi nuansa penting yang harus diperhitungkan pada analisis *trade-off*
performa-vs-efisiensi (§4.7): klaim efisiensi RM-b/RM-c berlaku kuat untuk *biaya
pelatihan* (relevan bagi skenario re-training berkala atau adaptasi ke domain baru), namun
**tidak serta-merta berlaku untuk biaya *serving*/inferensi** pada *deployment* volume
tinggi, di mana *latency* per prediksi menjadi metrik yang lebih kritis.
