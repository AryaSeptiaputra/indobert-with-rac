# RM-c — Rancangan Eksplorasi Rumus Fusi (133 konfigurasi per head)

Dokumen kerja untuk eksplorasi RM-c di `04_tuning_campaign.ipynb` (bagian 6),
pelengkap `RMC_TUNING_GRID.md`. RM-c standar menguji fusi linear di atas SATU head,
yaitu juara RM-b. Eksplorasi ini menguji RAC di atas SETIAP head RM-b dengan lima
rumus fusi, supaya klaim "RAC memperbaiki encoder beku" tidak bertumpu pada satu
titik operasi.

> **CSV yang tersedia:** `RMC_EXPLORATION_GRID.csv`, 133 baris, dimuat lewat
> `load_exploration_grid`. CSV hanya memuat sumbu fusi; sumbu head diambil saat
> runtime dari seluruh baris `runs_rmb.csv`.

---

## Sumbu

| Sumbu | Nilai | Catatan |
|---|---|---|
| Head | seluruh run di `runs_rmb.csv` | dimuat dari `checkpoints/rmb_heads/run_{id}.pt` |
| Rumus fusi | `linear`, `rumus1`, `rumus2`, `rumus3`, `rumus4` | `linear` adalah fusi RM-c produksi |
| alpha | 0,0 sampai 1,0, step 0,1 | hanya `linear` dan `rumus4`; rumus 1 sampai 3 tanpa alpha |
| k | 1, 3, 5, 10, 20, 50 | rentang sama dengan `RMC_TUNING_GRID.csv` |
| weighting | `similarity` dikunci | `uniform` dicek satu kali di sel juara |

Head dan fusi digrid bersama, bukan coordinate descent. Head menentukan seberapa
yakin `p_bert`, sehingga interaksinya dengan alpha nyata secara prinsip. Evaluasinya
juga murah: tidak ada pelatihan, dan pencarian tetangga dihitung sekali per split
lalu dipotong per k (`NeighborCache`).

## Isi grid

| Rumus | Konfigurasi | Baris |
|---|---|---|
| `linear` | alpha 0,0 (satu baris, k tidak berpengaruh) + alpha 0,1 sampai 1,0 x 6 k | 61 |
| `rumus4` | alpha 0,1 sampai 0,9 x 6 k | 54 |
| `rumus1` | 6 k | 6 |
| `rumus2` | 6 k | 6 |
| `rumus3` | 6 k | 6 |
| **Total** | | **133** |

Duplikat dibuang: untuk fusi linear, alpha=0 identik dengan head untuk semua k; untuk
Rumus 4, kedua ujung alpha identik dengan ujung fusi linear. Dengan 27 head, total
3.591 evaluasi.

## Rumus

Semua rumus memakai `p_bert = softmax(head(embedding))` dan retrieval train-only.

| Rumus | Definisi | Level |
|---|---|---|
| `linear` | `(1-alpha) * p_bert + alpha * p_retr` (`RACClassifier.fuse`) | probabilitas |
| `rumus1` | `(L/2) * (f_retr/‖f_retr‖ + logit/‖logit‖)`, Long dkk. (2022) | **skor** |
| `rumus2` | linear dengan `alpha` = rata-rata similarity k tetangga (per sampel) | probabilitas |
| `rumus3` | linear dengan `alpha = 1 - max(p_bert)` (per sampel) | probabilitas |
| `rumus4` | `p_bert^(1-alpha) * p_retr^alpha`, dinormalisasi ulang | probabilitas |

Rumus 1 tidak menghasilkan probabilitas (skornya bisa negatif) dan Rumus 4 memakai
pemangkatan, keduanya menyimpang dari aturan "softmax sekali, fusi level
probabilitas". Bila salah satunya juara, Bab 4 harus menyatakan rumusnya.

**Sitasi:** kode hanya mencantumkan Long dkk. (2022) untuk Rumus 1. Sumber Rumus 2,
3, dan 4 belum tercatat di kode dan perlu dilengkapi di sini sebelum dikutip di Bab 4.

## Aturan seleksi

1. Metrik utama `val_f1_macro`; pemecah seri `val_f1_judi`.
2. **Ambang seri 0,15 pp** (`settings.tie_threshold_pp`). Konfigurasi dalam ambang dari
   yang tertinggi dianggap seri, dan yang lebih murah menang, dengan urutan: parameter
   head lebih sedikit, waktu latih head lebih singkat (dibulatkan ke detik penuh agar
   derau pengukuran tidak menentukan), fusi `linear`, k lebih kecil, alpha lebih kecil.
3. **Penantang harus mengalahkan juara standar.** Ia menggantikan juara RM-c standar
   hanya bila selisih F1-macro-nya melampaui ambang seri DAN batas bawah interval
   bootstrap berpasangan 95% (2.000 resample, validation) berada di atas nol.
   Alasannya: dengan ribuan kandidat, pemenang mentah rawan bias seleksi. Pada sapuan
   fusi linear sebelumnya, kandidat teratas unggul 0,21 pp tetapi intervalnya
   [-0,50, +0,94] pp, artinya selisihnya tidak dapat dibedakan dari derau.
4. **Baca sebagai permukaan:** pivot alpha x k per rumus dan per head. Pemenang di
   tepi grid berarti rentang harus dilebarkan, bukan dikunci.
5. `alpha=0` adalah baseline wajib. Bila tidak ada kombinasi yang mengalahkannya
   secara berarti, itu temuan sah, bukan kegagalan.
6. Split test tidak disentuh sampai `05_final_benchmark.ipynb`.

## Tahap lanjutan (bergantung hasil)

`weighting=uniform` dicek satu kali di sel juara, dengan pola yang sama seperti
`RMC_TUNING_GRID_STAGE2.csv`. CSV-nya dibuat setelah eksplorasi selesai.

## Keluaran

Di `outputs/tuning/rmc_exploration/`:

| Berkas | Isi |
|---|---|
| `rmc_exploration_runs.csv` | satu baris per head x konfigurasi fusi, dengan baseline head dan `gain_pp` |
| `rmc_exploration_per_head.csv` | konfigurasi terbaik per head, biaya head, dan tanda Pareto |
| `rmc_exploration_per_formula.csv` | terbaik dan konfigurasi seragam terbaik per rumus |
| `champion_decision.json` | penantang, petahana, selisih, interval bootstrap, dan pemenang |

## Biaya

Tidak ada pelatihan; hitungan menit untuk 3.591 evaluasi. Batasan sebenarnya adalah
disiplin validation/test dan aturan seleksi di atas.
