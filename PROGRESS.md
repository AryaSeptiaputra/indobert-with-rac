# PROGRESS — IndoBERT-with-RAC

Status proyek skripsi ITENAS 2026 (trade-off performa versus efisiensi tiga
strategi adaptasi IndoBERT untuk deteksi komentar judi). Terakhir diperbarui:
**2026-09-02**.

## Ringkasan tahap

| Tahap | Status | Artefak |
|-------|--------|---------|
| EDA | Selesai | `docs/EDA_REPORT_BAGIAN1/2/3.md`, `outputs/figures/eda/` |
| Preprocessing | Selesai | `data/processed/`, `DATASET.md` |
| Penulisan ulang kode ke standar `writer-code` | Selesai | `src/`, `tests/`, `notebooks/` |
| Kampanye tuning lokal (RTX 3050) | **Belum dijalankan** | target `outputs/tuning/` |
| Benchmark final lokal (TEST, satu sesi) | **Belum dijalankan** | target `outputs/tuning/metrics/` |
| Penulisan Bab 4 | Belum | menunggu angka lokal |

## Keadaan sekarang

Kode telah ditulis ulang sepenuhnya (branch `rewrite/writer-code-standard`) dan
seluruh training dipindahkan dari Vast.ai ke mesin lokal. Kampanye lama di RTX
3090 **tidak lagi dipakai untuk Bab 4**; hasilnya disimpan di
`outputs/_archive_vast*/` sebagai jalan mundur sampai kampanye lokal terbukti
berhasil, lalu akan dihapus.

Yang sudah terverifikasi:

- Split `train/val/test` dan `data_clean.csv` yang dihasilkan kode baru
  **byte-identical** dengan yang dipakai seluruh eksperimen sebelumnya, dan
  seluruh field `metadata.json` sama persis (14.237 → 14.227 → 9.412 → 17 bocor
  dibuang → 9.395).
- `RACClassifier` menghasilkan prediksi identik dengan implementasi lama pada 30
  kombinasi alpha/k/weighting.
- `RMBTrainer` dan `RMCEvaluator` identik dengan implementasi lama, termasuk
  seluruh kurva `train_loss` per epoch.
- RM-a memuat **109.485.314** trainable parameter dan `MLPHead(768, 1024)`
  memuat **789.506**, keduanya cocok dengan angka yang dikutip sebelumnya.
- Smoke run ujung ke ujung di RTX 3050 (subset kecil): RM-a → RM-b → RM-c →
  benchmark final berjalan dan menghasilkan 19 figur, 3 tabel metrik, dan 3
  checkpoint.
- `pytest`: 243 test lulus, sekitar 50 detik, tanpa GPU.

## Dataset

`data/processed/{train,val,test}.csv` dengan kolom `textOriginal`, `text_clean`,
`label`. **Input model adalah `text_clean`.**

train 6.588 / val 1.402 / test 1.405 (total 9.395, sekitar 18% kelas judi, rasio
4,5:1). Class weight dari train `{0: 0.611, 1: 2.752}`. Metrik utama F1-macro.

Model dasar `indobenchmark/indobert-base-p2`, `max_length` 128, special token
`[URL]`, `[MENTION]`, `[NUM]`.

## Rencana kampanye lokal

**Hardware:** RTX 3050 Laptop, 4 GB VRAM (sekitar 3 GB bebas), 16 SM, 16 core CPU.

RM-a `batch=32` tidak muat di 3 GB, jadi dipakai `micro_batch=8` dengan akumulasi
gradien. Ini ekuivalen secara matematis dengan batch besar untuk BERT (LayerNorm,
bukan BatchNorm, dan loss dibagi jumlah akumulasi), sehingga merupakan kompromi
memori dan bukan kompromi hasil. Perlu disebutkan di Bab 4.

Perkiraan biaya, diskalakan dari kampanye 3090:

| Skenario | Run | Di 3090 | Perkiraan di 3050 |
|---|---|---|---|
| RM-a | 26 | 2.994 s | 4–6 jam |
| RM-b | 31 | 379 s | sekitar 35 menit |
| RM-c | 67 | 6 s | sekitar 1 menit |

Langkah pertama kampanye adalah kalibrasi satu run RM-a untuk mengukur waktu dan
memori sesungguhnya sebelum grid penuh dijalankan (sel pertama notebook 04).

## Yang akan berubah pada angka Bab 4

F1 tidak akan sama persis dengan hasil 3090: beda arsitektur GPU berarti beda
pemilihan kernel cuDNN dan beda urutan reduksi floating point, yang terakumulasi
selama lima epoch; ditambah transformers 5.12 versus 4.38. Perkirakan pergeseran
sekitar 0,1–0,5 pp, dan konfigurasi pemenang bisa berbeda. Narasi seperti "MLP
mengungguli linear" harus diperiksa ulang terhadap angka baru, bukan disalin.

Temuan yang diperkirakan bertahan karena bersifat arsitektural, bukan efek
hardware:

- Rasio waktu latih RM-a berbanding RM-b justru melebar di GPU yang lebih lemah,
  karena RM-b melatih head di atas fitur yang sudah dihitung.
- Latency inferensi RM-a, RM-b, dan RM-c berdekatan. Smoke run di RTX 3050
  memberi 73,1 / 71,2 / 73,0 ms per sampel — pola yang sama dengan 9,40 / 9,08 /
  10,96 ms di RTX 3090. Ketiganya menjalankan forward pass encoder 110 juta
  parameter yang sama, sehingga efisiensi RM-b dan RM-c ada pada parameter dan
  waktu LATIH, bukan pada kecepatan prediksi.

## Langkah berikutnya

1. Jalankan `04_tuning_campaign.ipynb` (mulai dari sel kalibrasi).
2. Jalankan `05_final_benchmark.ipynb` satu kali setelah ketiga skenario punya run.
3. Jalankan `06_analysis_export.ipynb` untuk biaya FAISS dan `HASIL.xlsx`.
4. Tulis Bab 4 dari angka di `outputs/tuning/`.
5. Setelah Bab 4 selesai, hapus `outputs/_archive_vast*/`.

## Aturan validitas Bab 4

Angka efisiensi (waktu latih, latency, peak memory) hanya sah bila berasal dari
satu hardware dan satu sesi. F1 tidak bergantung hardware.
