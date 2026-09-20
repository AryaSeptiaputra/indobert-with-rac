# PROGRESS — IndoBERT-with-RAC

Status proyek skripsi ITENAS 2026 (trade-off performa versus efisiensi tiga
strategi adaptasi IndoBERT untuk deteksi komentar judi). Terakhir diperbarui: **2026-09-20**.

Narasi branch `google-colab`: kampanye dijalankan di runtime Google Colab.

## Ringkasan tahap

| Tahap | Status | Artefak |
|-------|--------|---------|
| EDA | Selesai | `docs/EDA_REPORT_BAGIAN1/2/3.md`, `outputs/figures/eda/` |
| Preprocessing | Selesai | `data/processed/`, `DATASET.md` |
| Penulisan ulang kode ke standar `writer-code` | Selesai | `src/`, `tests/`, `notebooks/` |
| Kampanye tuning di Colab | Belum dimulai (GPU dan sesi diisi setelah `hardware.json` ada) | `outputs/tuning/` |
| Benchmark final di Colab (TEST, satu sesi) | Belum | `outputs/tuning/metrics/` |
| Penulisan Bab 4 | Belum | menunggu angka Colab |

## Keadaan sekarang

Branch `google-colab` dibuat pada 2026-09-20 dari branch `vast.ai` (commit `d42e71d`)
untuk kampanye di Google Colab. Codebase identik dengan branch lain; yang berbeda hanya
narasi ini. Hasil run tidak disimpan di git (lihat bagian di bawah): amankan dengan sel
"Arsipkan hasil" di `06_analysis_export.ipynb`, lalu unduh `hasil_*.tar.gz` atau salin
ke Google Drive sebelum runtime terputus. Cara memulai sesi Colab ada di `README.md`,
bagian "Mulai kerja di Google Colab".

Angka kampanye di branch `local` (RTX 3050) dan `vast.ai` (RTX 3090) tidak dipakai di
branch ini. Angka efisiensi hanya sah dari satu sesi dan satu tipe GPU; tipe GPU Colab
bisa berbeda antar sesi, jadi RM-a sampai `05_final_benchmark` dijalankan dalam satu
sesi.

Yang sudah terverifikasi di codebase (tidak bergantung hardware):

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
- `pytest`: 409 test lulus, tanpa GPU.

## Dataset

`data/processed/{train,val,test}.csv` dengan kolom `textOriginal`, `text_clean`,
`label`. **Input model adalah `text_clean`.**

train 6.588 / val 1.402 / test 1.405 (total 9.395, sekitar 18% kelas judi, rasio
4,5:1). Class weight dari train `{0: 0.611, 1: 2.752}`. Metrik utama F1-macro.

Model dasar `indobenchmark/indobert-base-p2`, `max_length` 128, special token
`[URL]`, `[MENTION]`, `[NUM]`.

## Kalibrasi biaya

Kalibrasi RTX 3050 dan perbandingannya dengan RTX 3090 ada di `PROGRESS.md` branch
`local`. Tidak dipakai di sini karena bergantung pada hardware lain. Biaya di Colab
diukur lewat sel "Kalibrasi biaya" di `04_tuning_campaign.ipynb`.

## Redesain alur RM-c (2026-09-19, dari branch `vast.ai`)

Kampanye RM-c dulu hanya menguji RAC di atas head juara RM-b. Alurnya sekarang dua
lapis, semuanya di `04_tuning_campaign.ipynb`:

1. **RM-c standar** (tidak berubah): fusi linear di atas head juara RM-b,
   `RMC_TUNING_GRID.csv`, 66 konfigurasi.
2. **Eksplorasi RM-c** (baru, bagian 6): SEMUA head RM-b x lima rumus fusi (linear
   produksi, Rumus 1 sampai 4) x k, `RMC_EXPLORATION_GRID.csv`, 133 konfigurasi per
   head (3.591 evaluasi untuk 27 head). Rancangannya di
   `tuning_grids/RMC_EXPLORATION_GRID.md`. Validation saja; test tidak dibuka.
3. **Putusan juara:** penantang hasil eksplorasi menggantikan juara standar hanya
   bila selisihnya melampaui ambang seri 0,15 pp DAN lolos bootstrap berpasangan.
   Bila menang, `rmc_best.pt` memuat head dan rumus fusinya, dan
   `05_final_benchmark.ipynb` memuat RM-c dari sana.

Notebook 03c2 dan 03c3 dilebur: 03c sekarang pengantar satu head (baseline, sweep alpha,
pembanding empat rumus fusi), setara pola 03a dan 03b. Tahap RM-b menyimpan state
SETIAP head di `checkpoints/rmb_heads/`. Biaya RM-c di `success_criteria` dan
`final_comparison` sekarang biaya head yang dipakainya, bukan nol.

**Status:** kode, test (409 lulus), grid, dan notebook sudah siap. Eksplorasi BELUM
dijalankan di kampanye resmi Colab. Untuk menjalankannya: jalankan `04_tuning_campaign.ipynb`
dari atas dalam satu sesi. Kampanye baru di runtime baru menjalankan RM-a, lalu RM-b (yang
menyimpan state setiap head di `checkpoints/rmb_heads/`), lalu RM-c standar dan eksplorasi
(`explore_rmc`, `decide_rmc_champion`). Sel `runner.restore_checkpoints(include_rma=True)`
hanya diperlukan bila `outputs/` dipulihkan dari arsip di sesi lanjutan, karena checkpoint
tidak ikut git dan menjalankan ulang skenario tidak membuat checkpoint juara kembali
(F1 yang sama tidak dipromosikan).

Uji kelayakan di mesin lokal (RTX 3050, folder sementara, BUKAN hasil resmi): eksplorasi
3.591 evaluasi selesai dalam 36 detik; fusi linear unggul di 25 dari 27 head; penantang
teratas (head run 25, linear alpha=0,4 k=1) selisih +0,11 pp dengan interval bootstrap
[-0,60, +0,81] pp, sehingga juara standar tetap.

## Gate reproduktibilitas split lintas lingkungan (2026-09-20)

Gate di `02_preprocessing.ipynb` dulu membandingkan SHA-256 BYTE berkas split, dan
menolak split yang sama di Vast.ai (`train: BERBEDA 3632f141... vs 24551e68...`).
Penyebabnya bukan perbedaan isi: pandas menulis ujung baris sesuai OS, dan git dengan
`core.autocrlf=true` di mesin Windows menormalkan CRLF menjadi LF saat commit,
termasuk CR yang ada di DALAM `textOriginal` (21 baris di train, 24 di seluruh
split; blob git 21 byte lebih pendek). Split yang dibangun di Linux membawa CR itu,
sedangkan blob-nya tidak.

Sekarang gate membandingkan checksum ISI kanonik (`src/utils/checksum.py`,
`DatasetBuilder.verify_reproducibility`): ujung baris di dalam sel diseragamkan ke
LF sebelum di-hash. Hasilnya sama untuk salinan kerja Windows, blob git, dan keluaran
Linux; isi, urutan baris, dan nama kolom yang benar-benar berubah tetap terdeteksi.
`write_csv` juga selalu menulis LF, dan sel Simpan tidak menulis ulang split yang
sudah identik. Checksum isi (16 karakter pertama): train `96775273eaa7`,
val `0e5d8c524d30`, test `ee3e2a7d8aa9`.

## Hasil run tidak lagi disimpan di git (2026-09-20)

`outputs/` sekarang di-gitignore seluruhnya, kecuali figur EDA dan README. Sebelumnya
`runs_*.csv`, `best.json`, `history/`, `metrics/`, dan figur ikut ter-commit. Di
instance baru itu membuat kampanye 04 gagal: `run_batch` melewati semua konfigurasi
yang sudah tercatat (resume) dan mengembalikan tabel kosong, sel `nlargest` lalu
crash, dan `BestTracker` tidak mempromosikan F1 yang sama sehingga checkpoint juara
tidak pernah dibuat. Sel 04 juga sekarang membaca riwayat dari `runs_frame`, bukan
dari nilai kembalian `run_batch`, supaya tetap jalan saat resume melewati semuanya.
Angka kampanye Vast.ai lama tetap ada di riwayat git (commit `c05cb7a`) dan di
`docs/`, dan ditandai tag `hasil-vast-2026-09-14` (commit `63392bd`, commit terakhir
yang masih memuat berkas hasil). Hasil kampanye berikutnya diamankan dengan sel
"Arsipkan hasil" di `06_analysis_export.ipynb`, yang membuat `hasil_<gpu>_<waktu>.tar.gz`
untuk diunduh sebelum instance dihancurkan.

## Langkah berikutnya

1. Di runtime Colab: siapkan lingkungan (`README.md`, "Mulai kerja di Google Colab"),
   pilih runtime GPU, lalu jalankan `02_preprocessing.ipynb` sampai gate checksum isi
   lolos.
2. Jalankan `04_tuning_campaign.ipynb` dari atas dalam satu sesi, lalu
   `05_final_benchmark.ipynb` pada sesi dan GPU yang sama.
3. Jalankan sel "Arsipkan hasil" di `06_analysis_export.ipynb` dan unduh
   `hasil_*.tar.gz` (atau salin ke Drive) sebelum runtime berakhir.
4. Tulis Bab 4 dari angka di `outputs/tuning/` (`final_comparison.csv`,
   `success_criteria.csv`, `tuning_summary.json`).

## Aturan validitas Bab 4

Angka efisiensi (waktu latih, latency, peak memory) hanya sah bila berasal dari
satu hardware dan satu sesi. F1 tidak bergantung hardware.
