# PROGRESS — IndoBERT-with-RAC

Status proyek skripsi ITENAS 2026 (trade-off performa versus efisiensi tiga
strategi adaptasi IndoBERT untuk deteksi komentar judi). Terakhir diperbarui: **2026-09-02**.

## Ringkasan tahap

| Tahap | Status | Artefak |
|-------|--------|---------|
| EDA | Selesai | `docs/EDA_REPORT_BAGIAN1/2/3.md`, `outputs/figures/eda/` |
| Preprocessing | Selesai | `data/processed/`, `DATASET.md` |
| Penulisan ulang kode ke standar `writer-code` | Selesai | `src/`, `tests/`, `notebooks/` |
| Kampanye tuning lokal (RTX 3050) | **Berjalan** (RM-a 1 dari 26 run) | `outputs/tuning/` |
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

## Kalibrasi kampanye lokal (2026-09-02)

**Hardware:** RTX 3050 Laptop, 4 GB VRAM, 16 SM, 16 core CPU.

Satu run RM-a penuh dengan baseline kanonik (`lr=2e-5, epochs=5, batch=16,
warmup=0,1, wd=0,01`) sudah dijalankan dan tercatat sebagai run #1 di
`outputs/tuning/`, sehingga bukan sekadar uji coba melainkan sel pertama grid.

| Ukuran | Nilai |
|---|---|
| val F1-macro | 0,974873 |
| val F1 judi | 0,958904 |
| Epoch terbaik | 3 dari 5 |
| Waktu latih | 1.101 s (18,4 menit) |
| Peak GPU memory | 2.339 MB dari 4.096 MB |
| micro_batch x akumulasi | 8 x 2 |
| Trainable params | 109.485.314 |

**Memori aman.** Sisa 1.757 MB pada `micro_batch=8`. Kampanye 3090 memakai
`micro_batch=16` dengan puncak 2.309 MB, jadi 16 pun muat di kartu ini dan
memberi utilisasi GPU lebih baik.

**Tetapi `micro_batch` bukan knob bebas.** Rumus gradien akumulasi memang
ekuivalen dengan batch besar, tetapi run-nya tidak: `DataLoader` dengan ukuran
batch berbeda mengonsumsi RNG berbeda, sehingga mask dropout dan komposisi batch
ikut berubah. Terbukti dari tiga run berkonfigurasi sama di riwayat: run #1 dan
#2 (`micro_batch` efektif 8) memberi kurva yang IDENTIK seluruhnya, sedangkan
run #3 (efektif 16) memberi kurva berbeda dan F1-macro 0,977266 versus 0,974873.
Artinya training deterministik terhadap seed, tetapi `micro_batch` adalah bagian
identitas konfigurasi. Grid RM-a mengunci `micro_batch = 32` (efektif 16 pada
batch 16), dan nilai itu tidak boleh berubah di tengah kampanye.

**Biaya sebenarnya jauh di atas perkiraan awal.** Estimasi 5x perlambatan
berdasarkan rasio SM dan bandwidth ternyata keliru: pengukuran memberi **7,7x**
untuk konfigurasi yang sama (143,2 s di 3090 versus 1.101 s di 3050). Proyeksi
yang berlaku:

| Skenario | Run | Perkiraan di 3050 |
|---|---|---|
| RM-a | 26 | sekitar 8 jam |
| RM-b | 31 | sekitar 30-45 menit |
| RM-c | 67 | beberapa menit |

Kampanye RM-a perlu dijalankan semalam, bukan beberapa jam.

### Perbandingan dengan run berkonfigurasi sama di RTX 3090

| | RTX 3090 (arsip) | RTX 3050 (lokal) |
|---|---|---|
| val F1-macro | 0,974873 | 0,974873 |
| val F1 judi | 0,958904 | 0,958904 |
| val accuracy | 0,985021 | 0,985021 |
| Epoch terbaik | 5 | 3 |
| Waktu latih | 143,2 s | 1.101,0 s |
| micro_batch x akumulasi | 16 x 1 | 8 x 2 |

Metrik puncaknya sama sampai enam desimal, tetapi ini **bukan** reproduksi
bit-identical: kurva per-epoch keduanya berbeda (mis. epoch 1 memberi F1-macro
0,9584 versus 0,9398), dan nilai terbaik itu kebetulan dicapai pada epoch yang
berbeda. Yang terjadi adalah kedua model, pada epoch terbaiknya masing-masing,
menghasilkan confusion matrix yang persis sama atas 1.402 sampel validation.

Implikasinya untuk Bab 4: pergeseran F1 antar hardware ternyata jauh lebih kecil
daripada dugaan awal (0,1-0,5 pp), tetapi konfigurasi pemenang tetap harus
ditentukan ulang dari kampanye lokal, bukan disalin dari arsip.

## Langkah berikutnya

1. Lanjutkan `04_tuning_campaign.ipynb` dari run #2 (run #1 sudah ada).
   Pertimbangkan menaikkan `MICRO_BATCH` ke 16 di `.env` untuk memangkas waktu.
2. Jalankan `05_final_benchmark.ipynb` satu kali setelah ketiga skenario punya run.
3. Jalankan `06_analysis_export.ipynb` untuk biaya FAISS dan `HASIL.xlsx`.
4. Tulis Bab 4 dari angka di `outputs/tuning/`.
5. Setelah Bab 4 selesai, hapus `outputs/_archive_vast*/`.

## Aturan validitas Bab 4

Angka efisiensi (waktu latih, latency, peak memory) hanya sah bila berasal dari
satu hardware dan satu sesi. F1 tidak bergantung hardware.
