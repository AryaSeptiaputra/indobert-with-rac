# RM-c — Rancangan Tuning Hyperparameter (seluruh head RM-b x 66 alpha x k)

Dokumen kerja untuk tuning RM-c (RAC — Retrieval-Augmented Classification) lewat
`03c_rmc_rac.ipynb`. Pola sama dengan `RMA_TUNING_GRID.md`/`RMB_TUNING_GRID.md`,
disesuaikan karena RM-c **tidak melatih apa pun** — hanya fusi probabilitas sebuah head
RM-b dengan distribusi hasil retrieval FAISS (`RMCEvaluator` di `src/services/training.py`;
index dibangun HANYA dari embedding train — anti-leakage, lihat `DATASET.md`).

> **Satu grid, satu rumus (2026-09-24).** Sumbu head digrid bersama `alpha × k`: setiap
> head yang tercatat di `runs_rmb.csv` (state-nya di `checkpoints/rmb_heads/`) dipasangkan
> dengan 66 konfigurasi di `RMC_TUNING_GRID.csv`, lewat `rmb_run_id` di `RMCConfig`.
> Rumus fusi hanya fusi linear di bawah. Pemisahan lama "RM-c standar" (head juara RM-b
> saja) dan "eksplorasi" (seluruh head, lima rumus, putusan bootstrap) sudah dilebur.
>
> **CSV yang tersedia:**
> - `RMC_TUNING_GRID.csv` — 66 baris grid `alpha × k` (weighting dikunci `similarity`).
>   CSV hanya memuat sumbu fusi; sumbu head dibentuk saat runtime di 03c dari
>   `runs_rmb.csv`, sehingga jumlah run = 66 × jumlah head.

---

## Mekanisme fusi (basis rancangan)

```python
p_final = (1 - alpha) * softmax(head(embedding)) + alpha * p_retrieval
```
(`rac.fuse`, `src/services/rac.py:96-98`) — `alpha=0` murni head RM-b, `alpha=1` murni retrieval
k-NN. `p_retrieval` dihitung dari `k` tetangga terdekat (cosine similarity, index FAISS
train-only) dengan bobot `similarity` (mirip cosine, default) atau `uniform` (voting rata,
`src/services/rac.py:59-74`).

**Karena tanpa training, RM-c jauh lebih murah bahkan dari RM-b** (hitungan
milidetik–detik/eval, tanpa forward pass BERT — cuma index search + aritmetika fusi) —
grid jauh lebih padat dari RM-a/RM-b bisa terjangkau. Tapi risiko *overfitting ke
validation set* (~1.402 sampel) **tetap ada** — bahkan lebih relevan di sini karena murni
memilih 2 angka (alpha, k) yang langsung memaksimalkan metrik val, tanpa proses training
di antaranya. Rentang & jumlah titik grid tetap dijaga proporsional (bukan sembarang padat
cuma karena murah), dan aturan tie-break/seleksi yang sama tetap berlaku ketat.

**`alpha` × `k` terkopel** — keduanya bersama menentukan seberapa besar & seberapa "encer"
pengaruh cabang retrieval terhadap prediksi akhir. **`weighting` relatif independen** dan
cuma berpengaruh kalau `alpha > 0` (di `alpha=0`, cabang retrieval sepenuhnya diabaikan,
apa pun `weighting`-nya) — jadi diuji terpisah di sel pemenang, bukan digrid bersama
(kalau digrid bersama, separuh dari 132 kombinasi jadi percuma di sekitar alpha kecil).

---

## Tahap 1 — Grid `alpha × k` (66 run, weighting=similarity dikunci)

- `alpha ∈ {0.0, 0.1, 0.2, ..., 1.0}` (11 titik, step 0,1) — mencakup dari murni head
  (`alpha=0`, setara RM-b sendiri — jadi baris ini sekaligus jadi pembanding langsung
  "apakah RAC benar memperbaiki RM-b?") sampai murni retrieval (`alpha=1`).
- `k ∈ {1, 3, 5, 10, 20, 50}` (6 titik) — rentang sama persis dengan grid ad-hoc yang
  sudah pernah dipakai di riwayat kampanye lama dan riwayat tuning RM-c sebelum
  clean-slate (dicatat `DATASET.md`: pemenang lama α=0,5 k=3 — **historis, hasil lama
  sudah dihapus, tak diasumsikan berulang**, sama prinsipnya dengan keputusan arsitektur
  RM-b).

11 × 6 = **66 kombinasi**. `weighting="similarity"` (default `RMC_DEFAULT`, `src/services/training.py:43`)
dikunci di seluruh sel — mendasarkan bobot tetangga pada cosine similarity, bukan voting
rata, sesuai desain default `rac.py`.

**Landasan:** Yu dkk. (2023) *"Retrieval-augmented few-shot text classification"*
(dikutip `src/services/training.py:42`) dan Long dkk. (2022) *"Retrieval augmented classification
for long-tail visual recognition"* — keduanya menunjukkan performa RAC sensitif terhadap
bobot fusi dan jumlah tetangga, memotivasi grid dua-sumbu ini alih-alih menerka satu nilai.
Chalkidis & Kementchedjhieva (2023) *"Retrieval-augmented multi-label text classification"*
jadi rujukan tambahan konteks task klasifikasi teks (bukan visual).

**Cara jalan:** unggah `RMC_TUNING_GRID.csv` lewat mode Batch → CSV.

---

## Tahap 2 — Cek `weighting=uniform` di sel juara (1 run, BERGANTUNG Tahap 1)

`weighting` hanya berpengaruh bila `alpha > 0` dan relatif independen dari `alpha × k`,
jadi diuji sekali di sel juara Tahap 1 (head, `alpha`, dan `k` yang sama), bukan digrid.
03c menyusun baris ini dari `best.json` setelah Tahap 1 selesai; tidak ada CSV terpisah.

> Catatan kampanye lama (historis, bukan acuan): di grid satu head, juara mekanis
> `alpha=0,2, k=1` seri secara statistik dengan `alpha=0,2, k=5` yang kurvanya jauh lebih
> mulus; kolom `k=1` identik dari `alpha=0,5` sampai `1,0`, ciri estimator varian tinggi.
> Pola seperti ini perlu dibaca lagi di tabel per head sebelum mengunci sel juara.

---

## Aturan seleksi (identik RM-a/RM-b)

1. **Metrik utama** `val_f1_macro`; **tie-break** `val_f1_judi`.
2. **Ambang seri ≤0,15pp** (`TIE_THRESHOLD_PP`) → kalau beberapa sel seri, pilih head
   dengan parameter lebih sedikit, lalu `alpha` lebih kecil (lebih dekat ke murni head
   RM-b) dan/atau `k` lebih kecil (retrieval lebih murah saat inferensi nyata).
   `best.json` memilih F1 tertinggi secara mekanis; kolom `is_tie_with_best` menandai
   kandidat seri yang perlu dibaca manual.
3. **Baca `alpha=0` sebagai baseline wajib** — kalau tak ada `alpha>0` yang mengalahkannya
   secara berarti, itu temuan valid (RAC tak membantu untuk kombinasi head/data ini),
   bukan kegagalan eksperimen.
4. **Baca grid sebagai permukaan**: pivot `alpha × k` (`reporting.AXES["rmc"] =
   ("alpha","k","weighting")` sudah otomatis membuat heatmap-nya) — cari apakah k optimal
   bergeser seiring alpha berubah.
5. **TEST tidak disentuh** sampai tab Final.

---

## Estimasi biaya

66 × jumlah head + 1 run (27 head = 1.783 run), semuanya sub-detik/eval (index FAISS
~6,6k vektor, tanpa training) — total kampanye **hitungan menit**. Bukan kendala biaya sama sekali; batasan
sebenarnya tetap disiplin val/test dan ambang tie-break di atas.

---

## Landasan literatur

| Sumber | Kontribusi ke rancangan |
|---|---|
| Yu dkk. (2023) | RAC few-shot: alpha membobot cabang parametrik vs retrieval → dasar sumbu `alpha` |
| Long dkk. (2022) | RAC long-tail: performa sensitif terhadap `k` tetangga → dasar sumbu `k` |
| Chalkidis & Kementchedjhieva (2023) | Retrieval-augmented text classification — konteks task teks (bukan visual) |

Prior-work tugas identik (deteksi judi Indonesia): Kamdan dkk. (2025), Manullang dkk.
(2025) — daftar lengkap `docs/DAFTAR_REFERENSI.pdf`.
