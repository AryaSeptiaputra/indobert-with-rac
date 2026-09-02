# Laporan EDA — Bagian 3: Keputusan Terpadu (Filtering + Transformasi)

| | |
|---|---|
| **Proyek** | Analisis Trade-off Performa & Efisiensi Adaptasi IndoBERT dengan RAC untuk Deteksi Komentar Promosi Judi Daring |
| **Penulis** | Arya Eka Septiaputra (152022190) — Informatika ITENAS 2026 |
| **Sumber data** | `data/raw/data_labeling.csv` |
| **Random seed** | 42 |
| **Notebook** | `notebooks/01_eda.ipynb` (Bagian 3) |
| **Sifat fase** | Keputusan — menetapkan spesifikasi final preprocessing |

> Bagian 3 mengambil seluruh keputusan penyaringan baris & transformasi teks berdasarkan angka **Bagian 1 (+susulan)** & **Bagian 2**. Filtering dan transformasi digabung karena saling memengaruhi (loop, bukan garis lurus — contoh: emoji removal → baris kosong → memaksa keputusan filtering baru). Output menjadi **spesifikasi langsung** `02_preprocessing.ipynb` dan bahan Bab 4. **Eksekusi nyata** (menulis file split) dilakukan di `02_preprocessing.ipynb`, bukan di sini.

---

## 1. Ringkasan Eksekutif

| Metrik | Nilai |
|--------|-------|
| Data awal | 14.237 |
| Dibuang — missing `textOriginal`/`label` | 10 |
| Dibuang — duplikat (kunci **NFKC-exact**) | 4.815 |
| Dibuang — kriteria lain (pendek/non-ID/outlier/emoji) | 0 (semua dipertahankan) |
| **Total dibuang** | **4.825** |
| **Data final** | **9.412** |
| Distribusi kelas final (L0 / L1) | 7.702 / 1.710 — **81,83% / 18,17%** |
| Rasio imbalance final | **4,50 : 1** (kelas 0 dominan) |
| NFKC (fondasi normalisasi) | **Dipakai** |
| Class weight pada loss | **Diperlukan** |
| Feasibility RM-b/RM-c | **Lolos** (train kelas 1 ~1.197; tipis, dipantau) |

**Tiga keputusan paling berdampak:**
1. **NFKC diadopsi** sebagai fondasi normalisasi (sebelum dedup & tokenisasi) — berdasar bukti `[UNK]` kelas 1 turun 90,5%→53,6% (Bagian 2).
2. **Dedup memakai kunci NFKC-exact** — menangkap varian Unicode kelas 1 (leakage-vector RM-c) tanpa over-merge.
3. **Kriteria filtering lain (pendek, non-ID, outlier, emoji) tidak diterapkan** — seluruhnya hanya akan memangkas kelas 0 dan/atau berisiko membuang data valid; imbalance ditangani via class weight, bukan pembuangan baris.

---

## 2. Prinsip Tetap (3A) — dari desain penelitian, tidak bergantung data

- **Dedup pada data penuh sebelum split** (anti-leakage; kritis untuk RM-c/FAISS).
- Exact duplicate `textOriginal` → **drop** (keep first).
- Missing `textOriginal` atau `label` → **drop**.
- Konflik label pada grup duplikat exact → default **assign label 1** (dikonfirmasi §1.8: mayoritas grup memang berlabel 1).
- URL / mention / angka → ganti **placeholder** (`[URL]`, `[MENTION]`, `[NUM]`), **bukan** dihapus — sinyal kuat promosi judi.
- Obfuskasi residual (`sl0t`, `g4cor`, small caps sisa NFKC) → **dokumentasikan saja**; tidak membangun normalizer leetspeak (future work).
- Tokenisasi memakai WordPiece bawaan IndoBERT **p2**.

## 3. Sudah Dikunci dari Bagian 1 (3B) — data-dependent, final

- Drop **10** baris `textOriginal` kosong (§1.2).
- `max_length = 128` token (P99=54; truncation 0,04%) (§1.4).
- Case folding manual **dilewati** (`do_lower_case=True`) (§1.4 / §2.4).
- **Stratified split 70/15/15**.

---

## 4. Keputusan Data-Dependent (3C) — Final + Justifikasi

| # | Keputusan | Ketetapan | Justifikasi (angka) |
|---|-----------|-----------|---------------------|
| 1 | **NFKC (fondasi normalisasi)** | **DIPAKAI** — sebelum dedup & tokenisasi | §2.2: baris L1 ber-`[UNK]` **90,5%→53,6%** (`[UNK]` L1 −40,1%). Dampak besar, biaya 1 pemanggilan standar. |
| 2 | **Kunci dedup** | **NFKC-exact** | §2.1: menangkap **+53** varian Unicode L1 (leakage-vector RM-c) tanpa over-merge. Full-norm (NFKC+dangkal, 5.253) di-*defer* karena lowercase/hapus-punct berisiko mengonflasi komentar pendek berbeda. |
| 3 | **Komentar terlalu pendek** | **Pertahankan** | §1.9: **seluruh** komentar pendek adalah L0; membuang hanya memangkas mayoritas & berisiko menghapus kampanye `stopjud*` (5 komentar valid). |
| 4 | **Emoji** | **Pertahankan** (tidak dihapus) | §2.3: emoji removal menciptakan **353** baris kosong (semua L0) → memaksa drop tambahan; tokenizer p2 sudah menangani emoji. |
| 5 | **Komentar non-Indonesia** | **Pertahankan** | §1.10: langdetect over-estimate (23%; 733 "tl" = Indonesia salah-deteksi); drop tak reliabel & berisiko buang 431 L1 + 2.864 L0. Dicatat sebagai limitation. |
| 6 | **Outlier panjang ekstrem** | **Pertahankan** | §1.11: top-20 terpanjang = diskusi esports valid (L0), bukan noise; `max_length=128` sudah menangani truncation. |
| 7 | **Normalisasi slang/leetspeak** | **Tidak dibangun** (future work) | §2.1: small caps & leetspeak residual bertahan setelah NFKC; robustness obfuskasi di luar scope. |
| 8 | **Class weight pada loss** | **Diperlukan** | §1.7: rasio pasca-dedup 4,37:1 (NFKC-key **4,50:1**), dari 1,51:1 mentah. Kombinasi F1-macro + stratified + class weight. |

**Feasibility gate RM-b/RM-c:** kelas 1 pasca-dedup ~1.710 (NFKC-key) → **train ~1.197** | val ~256 | test ~256. **Memadai** untuk RM-a (fine-tuning penuh) dan **cukup** untuk RM-b/RM-c (frozen encoder + head/retrieval butuh lebih sedikit data). **LOLOS**, dengan catatan kelas 1 tergolong tipis → pantau varians metrik, pertimbangkan class weight/threshold tuning. Untuk RM-c, index FAISS akan didominasi kelas 0 (4,5:1) → sensitivitas retrieval kelas minoritas perlu dimonitor saat evaluasi.

---

## 5. Tabel Before/After Count Final (3D)

Kunci dedup final = **NFKC-exact**. Semua keputusan filtering lain = pertahankan → tidak mengurangi baris.

| Tahap | Operasi | Baris |
|-------|---------|-------|
| Data awal | — | 14.237 |
| − Missing `textOriginal`/`label` | drop | −10 → 14.227 |
| − Duplikat (NFKC-exact) | drop keep-first | −4.815 → 9.412 |
| − Komentar pendek | pertahankan | −0 |
| − Non-Indonesia | pertahankan | −0 |
| − Outlier panjang | pertahankan | −0 |
| − Emoji-empty | emoji tidak dihapus | −0 |
| **Data final** | | **9.412** |

**Distribusi kelas final:** L0 = 7.702 (81,83%) · L1 = 1.710 (18,17%) · rasio **4,50:1**.

**Perbandingan kunci dedup (dokumentasi):**

| Kunci | Final | L0 | L1 | Catatan |
|-------|-------|-----|-----|---------|
| exact | 9.465 | 7.702 | 1.763 | baseline; buta varian Unicode |
| **NFKC-exact (dipilih)** | **9.412** | **7.702** | **1.710** | menangkap +53 near-dup Unicode L1 |
| NFKC+dangkal (defer) | 8.974 | 7.335 | 1.639 | agresif; risiko over-merge |

---

## 6. Ringkasan Keputusan (spesifikasi `02_preprocessing.ipynb`)

| Kriteria | Keputusan | Threshold / Kunci | Baris Dibuang |
|----------|-----------|-------------------|---------------|
| Missing `textOriginal`/`label` | Drop | — | 10 |
| Normalisasi NFKC (fondasi) | **Dipakai** | `NFKC` sebelum dedup & tokenisasi | (bukan drop) |
| Dedup `textOriginal` | Drop (keep first) | **NFKC-exact** | 4.815 |
| Komentar terlalu pendek | **Pertahankan** | — | 0 |
| Komentar non-Indonesia | **Pertahankan** | — | 0 |
| Outlier panjang ekstrem | **Pertahankan** | — | 0 |
| Emoji | **Pertahankan** | — | 0 |
| URL/mention/angka | Ganti placeholder | `[URL]`/`[MENTION]`/`[NUM]` | (transformasi) |
| Case folding | Dilewati | `do_lower_case=True` | — |
| Konflik label duplikat | Assign label 1 | — | (bukan drop) |
| **Total dibuang** | | | **4.825** |
| **Data final** | | | **9.412** |
| **Distribusi kelas final** | L0=7.702 / L1=1.710 | rasio **4,50:1** | |
| **Class weight** | **Diperlukan** | F1-macro + stratified + class weight | |
| **Feasibility RM-b/RM-c** | **Lolos** (train L1 ~1.197) | | |

> Random seed **42** (sampling manual & split). Split stratified 70/15/15. Index FAISS RM-c dibangun **hanya dari train set** (prinsip anti-leakage non-negotiable).

---

*Laporan ini adalah keluaran final fase EDA: spesifikasi preprocessing yang langsung dieksekusi di `02_preprocessing.ipynb`, sekaligus bahan Bab 4 (Hasil & Pembahasan).*
