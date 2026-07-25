# RM-a — Rancangan Eksplorasi Hyperparameter (26 run)

Dokumen kerja untuk tuning RM-a (full fine-tuning IndoBERT) di tab **Tuning** pada `app.py`.
Isi kolom **Alasan** langsung dapat disalin ke field `catatan` di UI.

> **`RMA_TUNING_GRID.csv`** (folder ini, `tuning_grids/`) adalah transkripsi mesin-terbaca dari ke-24 baris
> Tahap 1 di bawah (kolom `lr,epochs,batch,warmup_ratio,weight_decay,micro_batch,catatan`).
> Unggah langsung lewat mode **Batch → Tempel/unggah tabel CSV** di tab Tuning `app.py`
> untuk menjalankan seluruh Tahap 1 dalam satu aksi — lihat `VAST_GUIDE.md` §4 "Mode Batch".
> Tabel markdown di bawah ini tetap jadi **sumber kebenaran**; jika tabel berubah, salin
> ulang manual ke CSV. Tahap 2 (2 baris coordinate-descent) SENGAJA tidak ada di CSV karena
> bergantung pada sel pemenang Tahap 1 — isi manual lewat tombol "Isi dari config terbaik
> saat ini" setelah Tahap 1 selesai.

---

## Metode: hibrida (grid + coordinate descent)

Partisi axis diturunkan dari mekanisme nyata di `src/tuning.py:108-114`, bukan asumsi:

```python
steps = ceil(len(tr)/accum) * epochs          # = ceil(N/batch) × epochs
sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * steps), steps)
opt   = AdamW(..., lr=lr, weight_decay=wd)
```

- **Kelompok A — terkopel kuat → grid kombinatorial:** `lr` × `epochs` × `batch`.
  Ketiganya bersama mendefinisikan lintasan optimasi: jumlah update = `ceil(N/batch) × epochs`,
  besar langkah = `lr`, horizon peluruhan = `steps`. Mengubah satu menggeser optimum dua lainnya.
- **Kelompok B — relatif independen → coordinate descent:** `warmup_ratio`, `weight_decay`.
  `warmup_ratio` berbasis **rasio** sehingga menormalkan diri terhadap batch/epochs;
  `weight_decay` di AdamW *decoupled by design* (Loshchilov & Hutter).

Grid 5-axis penuh (96 run, ~7 jam) **ditolak**: memakan hampir seluruh budget $2 dan
menaikkan risiko *overfitting ke validation set* (~1.400 sampel — makin banyak konfigurasi
dibandingkan, makin besar peluang pemenang menang karena keberuntungan).

**Ukuran train:** N = 6.588 (`dataset/processed/metadata.json`) →
`ceil(6588/16) = 412` step/epoch · `ceil(6588/32) = 206` step/epoch.

---

## Tahap 1 — Grid kombinatorial (24 run)

**Dikunci untuk SEMUA sel:** `warmup_ratio = 0.1`, `weight_decay = 0.01`, `micro_batch = 32`, `seed = 42`

| # | lr | epochs | batch | warmup | wd | update | Alasan kombinasi ini diuji |
|---|-----|--------|-------|--------|------|--------|-----------------------------|
| 1 ★ | 2e-5 | 5 | 16 | 0.1 | 0.01 | 2.060 | Baseline kanonik IndoNLU/Devlin — titik acuan seluruh grid |
| 2 | 2e-5 | 3 | 16 | 0.1 | 0.01 | 1.236 | Apakah lr kanonik cukup dengan anggaran lebih pendek? Cek underfit |
| 3 | 2e-5 | 8 | 16 | 0.1 | 0.01 | 3.296 | Anggaran terbesar pada lr kanonik — masih naik atau mulai overfit? |
| 4 | 1e-5 | 5 | 16 | 0.1 | 0.01 | 2.060 | lr konservatif, anggaran standar: kestabilan (Sun 2019) vs konvergensi lambat |
| 5 | 1e-5 | 3 | 16 | 0.1 | 0.01 | 1.236 | Kandidat underfit di batch 16 — batas bawah yang berguna |
| 6 | 1e-5 | 8 | 16 | 0.1 | 0.01 | 3.296 | lr kecil dikompensasi anggaran terbesar — cukupkah bila diberi waktu? |
| 7 | 3e-5 | 5 | 16 | 0.1 | 0.01 | 2.060 | lr tengah GLUE, anggaran standar; kandidat kuat bila 2e-5 terlalu lambat |
| 8 | 3e-5 | 3 | 16 | 0.1 | 0.01 | 1.236 | Konvergensi cepat menghemat waktu tanpa rugi F1? |
| 9 | 3e-5 | 8 | 16 | 0.1 | 0.01 | 3.296 | lr tengah + anggaran maksimum — titik rawan overfit |
| 10 | 5e-5 | 5 | 16 | 0.1 | 0.01 | 2.060 | Batas atas GLUE — uji ambang catastrophic forgetting |
| 11 | 5e-5 | 3 | 16 | 0.1 | 0.01 | 1.236 | Skenario "cepat & panas" — paling hemat waktu bila berhasil |
| 12 | 5e-5 | 8 | 16 | 0.1 | 0.01 | 3.296 | Paling berisiko merusak bobot pra-latih — batas atas |
| 13 | 2e-5 | 5 | 32 | 0.1 | 0.01 | 1.030 | **Pasangan langsung #1, batch digandakan — inti uji interaksi lr↔batch** |
| 14 | 2e-5 | 3 | 32 | 0.1 | 0.01 | 618 | Anggaran paling minim di grid — batas bawah ekstrem |
| 15 | 2e-5 | 8 | 32 | 0.1 | 0.01 | 1.648 | Batch besar + epoch terbanyak; gradien lebih halus dari #1 |
| 16 | 1e-5 | 5 | 32 | 0.1 | 0.01 | 1.030 | lr terkecil + update sedikit — kandidat underfit terkuat |
| 17 | 1e-5 | 3 | 32 | 0.1 | 0.01 | 618 | Sel paling konservatif — jangkar batas bawah |
| 18 | 1e-5 | 8 | 32 | 0.1 | 0.01 | 1.648 | Cukupkah epoch maksimum mengompensasi lr rendah + batch besar? |
| 19 | 3e-5 | 5 | 32 | 0.1 | 0.01 | 1.030 | **Uji linear scaling rule: 3e-5 kalahkan 2e-5 saat batch digandakan?** |
| 20 | 3e-5 | 3 | 32 | 0.1 | 0.01 | 618 | lr tengah + anggaran minim — efisiensi ekstrem |
| 21 | 3e-5 | 8 | 32 | 0.1 | 0.01 | 1.648 | lr tengah + batch besar + epoch maksimum — kandidat seimbang |
| 22 | 5e-5 | 5 | 32 | 0.1 | 0.01 | 1.030 | **Prediksi linear scaling: lr tinggi paling masuk akal di batch besar** |
| 23 | 5e-5 | 3 | 32 | 0.1 | 0.01 | 618 | lr agresif + update paling sedikit — batas forgetting anggaran minim |
| 24 | 5e-5 | 8 | 32 | 0.1 | 0.01 | 1.648 | Sudut grid berlawanan dari #17 — lr maksimum, anggaran besar |

★ = baseline kanonik IndoNLU/Wilie (2020), **dijalankan pertama** sebagai sanity check.

**Pasangan diagnostik:** #13 (1.030 update) vs #2 (1.236) — anggaran update hampir sama tapi
batch berbeda → memisahkan efek *jumlah update* dari efek *noise gradien*.

---

## Tahap 2 — Coordinate descent di sel pemenang (2 run)

Dijalankan setelah grid selesai, memakai `(lr*, ep*, b*)` = sel dengan val F1-macro terbaik.

| # | lr | epochs | batch | warmup | wd | Alasan kombinasi ini diuji |
|---|-----|--------|-------|--------|------|-----------------------------|
| 25 | lr* | ep* | b* | **0.0** | 0.01 | Apakah warmup memang perlu? Diuji **pada lr pemenang** → sekaligus menutup risiko sisa interaksi lr↔warmup |
| 26 | lr* | ep* | b* | 0.1 | **0.1** | Regularisasi lebih kuat — relevan bila grid menunjukkan gejala overfit |

---

## Template `catatan` untuk UI

```
Grid kombinatorial RM-a, sel lr=[X] / epochs=[Y] / batch=[Z]; warmup 0.1 & wd 0.01
dikunci. Tujuan: memetakan interaksi lr×epochs×batch (jumlah update =
ceil(N/batch)×epochs) yang tak tertangkap coordinate descent. [Peran sel].
```

Isi `[Peran sel]` dari kolom **Alasan** di tabel. Contoh:
- **#1** → `Baseline kanonik IndoNLU/Devlin — titik acuan seluruh grid`
- **#6** → `lr rendah + anggaran update penuh: uji apakah kestabilan (Sun 2019) menang atas konvergensi lambat`
- **#23** → `lr agresif + update paling sedikit: uji batas catastrophic forgetting`

---

## Aturan seleksi

1. **Metrik utama** `val_f1_macro`; **tie-break** `val_f1_judi` (F1 kelas-1/judi — kelas
   minoritas yang bisa tertutup oleh rata-rata macro).
2. **Ambang noise 0,1–0,2pp**: selisih sekecil itu dianggap **seri** (seed tunggal 42) →
   pilih konfigurasi yang lebih murah (epoch lebih kecil / batch lebih besar).
3. **Baca grid sebagai permukaan, bukan daftar**: susun pivot `lr × epochs` terpisah per
   `batch` untuk melihat apakah lr optimal benar-benar bergeser saat batch berubah.
4. **Waspadai pemenang di tepi grid** (mis. lr=5e-5 atau epochs=8) → sinyal optimum mungkin
   di luar rentang; pertimbangkan 1–2 run perluasan sebelum mengunci.
5. **Bandingkan efek utama baris/kolom**, jangan hanya sel individual — sel juara yang
   tetangganya jelek biasanya noise.
6. **TEST tidak disentuh** sampai tab Final. Biarkan "Evaluasi TEST juga" tidak dicentang.

---

## Estimasi biaya

128 epoch total (batch 16 ~50 dtk/epoch, batch 32 ~35 dtk/epoch) ≈ **1,6–1,8 jam ≈ $0,40**
di RTX 3090 — menyisakan anggaran cukup untuk RM-b, RM-c, dan tab Final.

Jalankan **1× Mode SMOKE** sebelum run penuh untuk memastikan pipeline menulis baris ke
`results/vast/runs_rma.csv` tanpa error.

---

## Landasan literatur

| Sumber | Kontribusi ke rancangan |
|---|---|
| Devlin dkk. (2019) | Grid GLUE: lr {2e-5, 3e-5, 5e-5}, batch {16, 32}, 2–4 epoch, warmup 0.1 |
| Wilie dkk. (2020) — IndoNLU | Config kanonik IndoBERT: lr ≈2e-5, batch 16, 5 epoch + early stopping → baseline #1 |
| Sun dkk. (2019) | Rekomendasi base lr 2e-5; peringatan lr terlalu tinggi → *catastrophic forgetting* |
| Loshchilov & Hutter | AdamW: weight decay *decoupled* → dasar memasukkan `wd` ke Kelompok B |

Prior-work tugas identik (deteksi judi Indonesia) yang relevan untuk pembanding HP:
Kamdan dkk. (2025), Manullang dkk. (2025, JAIC), Amin dkk. (2024) — daftar lengkap di
`DAFTAR_REFERENSI.pdf`.
