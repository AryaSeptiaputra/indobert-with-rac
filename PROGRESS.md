# PROGRESS — IndoBERT-with-RAC

Status proyek skripsi ITENAS 2026 (trade-off performa versus efisiensi tiga
strategi adaptasi IndoBERT untuk deteksi komentar judi). Terakhir diperbarui: **2026-09-24**.

## Ringkasan tahap

| Tahap | Status | Artefak |
|-------|--------|---------|
| EDA | Selesai | `docs/EDA_REPORT_BAGIAN1/2/3.md`, `outputs/figures/eda/` |
| Preprocessing | Selesai | `data/processed/`, `DATASET.md` |
| Penulisan ulang kode ke standar `writer-code` | Selesai | `src/`, `tests/`, `notebooks/` |
| Restrukturisasi notebook (03a-c jadi kampanye penuh, 04 rencana jadi figur) | Selesai | `notebooks/03a_rma_finetune.ipynb`, `03b_rmb_frozen.ipynb`, `03c_rmc_rac.ipynb` |
| Kampanye tuning (RM-a/RM-b/RM-c standar+eksplorasi) | Belum, menunggu rerun di struktur baru | `outputs/tuning/` (kosong) |
| Benchmark final (TEST, satu sesi) | Belum | `outputs/tuning/metrics/` |
| Penulisan Bab 4 | Belum | menunggu angka kampanye baru |

**2026-09-24:** `03a_rma_finetune.ipynb`, `03b_rmb_frozen.ipynb`, dan
`03c_rmc_rac.ipynb` diubah dari pengantar satu-konfigurasi menjadi kampanye
penuh per skenario (fungsi yang sebelumnya di `04_tuning_campaign.ipynb`
sekarang berjalan di masing-masing notebook 03, dengan keluaran ke
`outputs/tuning/` yang dipakai bersama, bukan lagi `outputs/baseline/`). 03c
menjalankan RM-c standar (`RMC_TUNING_GRID*.csv`, 66 konfigurasi alpha x k)
maupun eksplorasi (`RMC_EXPLORATION_GRID.csv`, 133 konfigurasi x seluruh head
RM-b) plus putusan juara. `04_tuning_campaign.ipynb` akan dialihfungsikan
menjadi notebook pembangkit figur untuk jurnal/skripsi (rancangan figur belum
diputuskan; kontennya masih kampanye lama untuk sementara). Angka kampanye
lama di ringkasan bawah ini (RTX 3050, 2026-09-02) sudah TIDAK berlaku untuk
Bab 4 karena strukturnya berubah; `outputs/tuning/` di working tree ini kosong
dan belum ada hasil run yang tersimpan di git pada branch `vast.ai` — kampanye
perlu dijalankan ulang dari `03a` untuk menghasilkan keluaran baru.

## Keadaan sekarang

Kode telah ditulis ulang sepenuhnya (branch `rewrite/writer-code-standard`) dan
seluruh training dipindahkan dari Vast.ai ke mesin lokal. Kampanye lama di RTX
3090 **tidak lagi dipakai untuk Bab 4**. Kampanye lokal (RTX 3050) sudah selesai
dan terverifikasi bebas cacat (2026-09-05): RM-a 30 run, RM-b 27 run, RM-c 67
run, tanpa error/NaN/duplikat, kriteria sukses RM-b dan RM-c 3/3 terpenuhi.
`outputs/_archive_vast*/` dan `outputs/combined/` (turunan arsip) sudah
dihapus.

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
- `pytest`: 409 test lulus, tanpa GPU.

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

## Redesain alur RM-c (2026-09-19, branch `vast.ai`)

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
dijalankan di kampanye resmi; angka Bab 4 tetap yang di `best.json`. Untuk menjalankan
di kampanye Vast.ai yang ada: jalankan `04_tuning_campaign.ipynb` dari atas. Sel
`runner.restore_checkpoints(include_rma=True)` (sebelum bagian 5) memulihkan head RM-b,
`rmb_best.pt`, `rmc_best.pt`, dan `rma_best.pt` dari konfigurasi di `best.json`; di RTX
3090 yang sama dengan kampanye asli hasilnya seharusnya identik, dan peringatan muncul
bila selisih > 0,05 pp dari catatan. Lalu `explore_rmc` dan `decide_rmc_champion`.
Checkpoint tidak ikut git, dan menjalankan ulang skenario tidak membuat checkpoint
juara kembali karena F1 yang sama tidak dipromosikan (2026-09-20: 03c gagal dengan
`rmb_best.pt tidak ada` di instance baru).

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

1. Tulis Bab 4 dari angka di `outputs/tuning/` (`final_comparison.csv`,
   `success_criteria.csv`, `tuning_summary.json`).

## Aturan validitas Bab 4

Angka efisiensi (waktu latih, latency, peak memory) hanya sah bila berasal dari
satu hardware dan satu sesi. F1 tidak bergantung hardware.
