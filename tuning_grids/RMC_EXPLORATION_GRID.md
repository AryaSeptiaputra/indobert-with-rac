# RM-c — Rancangan Eksplorasi Seluruh Head RM-b (61 konfigurasi per head)

Dokumen kerja untuk eksplorasi RM-c di `03c_rmc_rac.ipynb` (bagian 4),
pelengkap `RMC_TUNING_GRID.md`. RM-c standar menguji fusi linear di atas SATU head,
yaitu juara RM-b. Eksplorasi ini menguji fusi yang SAMA di atas SETIAP head RM-b,
supaya klaim "RAC memperbaiki encoder beku" tidak bertumpu pada satu titik operasi.

> **CSV yang tersedia:** `RMC_EXPLORATION_GRID.csv`, 61 baris, dimuat lewat
> `load_exploration_grid`. CSV hanya memuat sumbu fusi; sumbu head diambil saat
> runtime dari seluruh baris `runs_rmb.csv`.

---

## Rumus

Hanya satu rumus: fusi linear-konveks produksi (`RACClassifier.fuse`), yaitu rumus
RM-c sejak awal.

```
p_bert  = softmax(head(embedding))
p_retr  = distribusi label k tetangga terdekat (indeks FAISS hanya dari train)
p_final = (1 - alpha) * p_bert + alpha * p_retr
```

Fusi di level probabilitas, softmax sekali di cabang BERT sebelum fusi.

Empat rumus alternatif (Rumus 1 sampai 4 di `src/services/fusion_ablation.py`)
sempat masuk grid ini dan sudah dikeluarkan (2026-09-24); kodenya masih ada tetapi
tidak dipakai kampanye.

## Sumbu

| Sumbu | Nilai | Catatan |
|---|---|---|
| Head | seluruh run di `runs_rmb.csv` | dimuat dari `checkpoints/rmb_heads/run_{id}.pt` |
| alpha | 0,0 sampai 1,0, step 0,1 | sama dengan `RMC_TUNING_GRID.csv` |
| k | 1, 3, 5, 10, 20, 50 | sama dengan `RMC_TUNING_GRID.csv` |
| weighting | `similarity` dikunci | `uniform` dicek satu kali di sel juara |

Ruang alpha x k identik dengan grid standar (66 konfigurasi). Grid ini memuat 61
baris karena alpha=0 identik dengan head itu sendiri untuk semua k, sehingga cukup
satu baris (alpha 0,0 + 10 alpha x 6 k = 61). Dengan 27 head, total 1.647 evaluasi.

Head dan fusi digrid bersama, bukan coordinate descent. Head menentukan seberapa
yakin `p_bert`, sehingga interaksinya dengan alpha nyata secara prinsip. Evaluasinya
juga murah: tidak ada pelatihan, dan pencarian tetangga dihitung sekali per split
lalu dipotong per k (`NeighborCache`).

## Aturan seleksi

1. Metrik utama `val_f1_macro`; pemecah seri `val_f1_judi`.
2. **Ambang seri 0,15 pp** (`settings.tie_threshold_pp`). Konfigurasi dalam ambang dari
   yang tertinggi dianggap seri, dan yang lebih murah menang, dengan urutan: parameter
   head lebih sedikit, waktu latih head lebih singkat (dibulatkan ke detik penuh agar
   derau pengukuran tidak menentukan), k lebih kecil, alpha lebih kecil.
3. **Penantang harus mengalahkan juara standar.** Ia menggantikan juara RM-c standar
   hanya bila selisih F1-macro-nya melampaui ambang seri DAN batas bawah interval
   bootstrap berpasangan 95% (2.000 resample, validation) berada di atas nol.
   Alasannya: dengan ratusan kandidat, pemenang mentah rawan bias seleksi. Pada sapuan
   fusi linear sebelumnya, kandidat teratas unggul 0,21 pp tetapi intervalnya
   [-0,50, +0,94] pp, artinya selisihnya tidak dapat dibedakan dari derau.
4. **Baca sebagai permukaan:** pivot alpha x k per head. Pemenang di tepi grid berarti
   rentang harus dilebarkan, bukan dikunci.
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
| `rmc_exploration_per_formula.csv` | satu baris (`linear`): terbaik dan konfigurasi seragam terbaik |
| `champion_decision.json` | penantang, petahana, selisih, interval bootstrap, dan pemenang |

## Biaya

Tidak ada pelatihan; hitungan menit untuk seluruh evaluasi. Batasan sebenarnya adalah
disiplin validation/test dan aturan seleksi di atas.
