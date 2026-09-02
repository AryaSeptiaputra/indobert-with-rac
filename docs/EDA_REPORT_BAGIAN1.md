# Laporan EDA — Bagian 1: Eksplorasi Data General

| | |
|---|---|
| **Proyek** | Analisis Trade-off Performa & Efisiensi Adaptasi IndoBERT dengan RAC untuk Deteksi Komentar Promosi Judi Daring |
| **Penulis** | Arya Eka Septiaputra (152022190) — Informatika ITENAS 2026 |
| **Sumber data** | `data/raw/data_labeling.csv` |
| **Random seed** | 42 |
| **Notebook** | `notebooks/01_eda.ipynb` (Bagian 1) |
| **Sifat fase** | Deskriptif — **tidak ada baris yang diubah, dibuang, atau disimpan** |

> Laporan ini memotret kondisi data apa adanya dan menarik implikasinya untuk pipeline. Seluruh **keputusan** penyaringan dan transformasi ditetapkan di Bagian 2–3 dan dieksekusi di `02_preprocessing.ipynb`.

---

## 1. Ringkasan Eksekutif

**Angka pokok:**

| Metrik | Nilai |
|--------|-------|
| Total baris | 14.237 |
| Kolom | 11 (relevan: `textOriginal`, `label`; sisanya metadata) |
| Distribusi label (0 / 1) | 8.560 / 5.677 — **60,13% / 39,87%** |
| Rasio imbalance (mentah) | 1,51 : 1 (mayoritas kelas 0) |
| Missing `textOriginal` | 10 |
| Missing `label` | 0 |
| Duplikat `commentId` | 0 |
| Duplikat `textOriginal` exact | 5.922 baris terlibat (**4.771 akan dibuang**) |
| Near-duplicate (estimasi dangkal) | 6.368 baris (44,73%) — *upper bound, lihat caveat §5* |
| Median panjang token (kelas 0 / 1) | 9 / 13 token |
| P95 / P99 panjang token | 32 / 54 token |
| Komentar > 128 token | 0,04% |
| `do_lower_case` tokenizer | `True` |
| Rentang waktu | 2025-07-31 s/d 2025-10-01 (≈ 62 hari) |

**Empat temuan utama:**

1. **Duplikasi masif dan terkonsentrasi di kelas 1.** Dari 5.922 baris ber-duplikat exact, **4.850 (82%) berlabel 1**. Karena dedup wajib dilakukan sebelum split (anti-leakage), penghapusan ini akan **memangkas kelas 1 jauh lebih dalam daripada kelas 0**, dan diperkirakan **membalik tingkat imbalance** dari ringan (1,5:1) menjadi cukup berat (≈4–4,5:1). Ini implikasi terbesar fase ini (§9.1).

2. **Obfuskasi label 1 didominasi manipulasi Unicode, bukan leetspeak.** Nama merek judi disamarkan dengan *mathematical bold/italic*, *fullwidth*, *small caps*, *enclosed/negative-squared*, dan *variation selector* (mis. `ℙℝ𝕆𝔹𝔼𝕋855`, `🅿︎U︎L︎A︎U︎777`, `ＰＵＬＡＵＷＩＮ`). Ini membuka peluang normalisasi murah berdampak besar (Unicode NFKC) yang perlu dievaluasi di Bagian 2–3 (§9.5).

3. **Estimasi near-duplicate 44,73% kemungkinan besar terlalu rendah.** Fungsi normalisasi saat ini tidak menyentuh varian Unicode, sehingga dua komentar yang secara semantik identik tetapi beda byte (mis. `𝐏𝐑𝐎𝐁𝐄𝐓855` vs `ℙℝ𝕆𝔹𝔼𝕋855`) tidak terdeteksi sebagai near-dup (§5).

4. **`max_length = 128` terkonfirmasi sangat aman.** Hanya 0,04% komentar melebihi 128 token; P99 hanya 54 token. Truncation praktis tidak menghilangkan informasi.

---

## 2. Profil Dataset (1.0)

14.237 baris × 11 kolom. Hanya dua kolom yang dipakai pemodelan:

- **`textOriginal`** — teks komentar mentah (fitur)
- **`label`** — 0 (non-judi) / 1 (promosi judi) (target)

Sisanya metadata yang akan didrop saat preprocessing: `number`, `commentId`, `author`, `authorChannelId`, `textDisplay`, `likeCount`, `publishedAt`, `updatedAt`, `comment_id`.

> Catatan: `textDisplay` (versi ber-HTML, mis. `&quot;`, `<br>`) berbeda dari `textOriginal` (versi sudah ter-unescape). Pemodelan memakai `textOriginal`, sehingga beban pembersihan HTML lebih ringan — konfirmasi di Bagian 3.

---

## 3. Distribusi Label (1.1)

![Distribusi Label](outputs/figures/eda/label_distribution.png)

- Kelas 0 (non-judi): **8.560 (60,13%)**
- Kelas 1 (judi): **5.677 (39,87%)**
- Rasio **1,51 : 1** — imbalance ringan **pada data mentah**.

**Implikasi:**

- Imbalance ringan ini **menyesatkan** jika dibaca tanpa konteks dedup. Lihat §9.1 — setelah deduplikasi, rasio diperkirakan memburuk signifikan.
- **F1-macro** tetap menjadi metrik utama (sesuai proposal), bukan accuracy.
- **Stratified split** wajib agar proporsi kelas terjaga di train/val/test.
- Untuk **RM-c**, komposisi kelas memengaruhi isi vector database FAISS; dominasi satu kelas akan membiaskan distribusi k-NN.

---

## 4. Missing Values (1.2)

| Kolom | Jumlah NaN |
|-------|-----------|
| `textOriginal` | 10 |
| `label` | 0 |
| (kolom lain) | 0 |

Hanya 10 baris `textOriginal` kosong, dan **label lengkap 100%**. Sepuluh baris ini masuk kategori **drop pasti** di Bagian 2 (tidak ada teks untuk diklasifikasi). Skalanya sangat kecil (0,07%) sehingga tidak berdampak ke distribusi.

---

## 5. Duplikasi (1.3) — Temuan Paling Kritis

| Jenis | Jumlah | Catatan |
|-------|--------|---------|
| (a) Duplikat `commentId` | 0 | Tidak ada artefak pengambilan ganda dari API |
| (b) Duplikat `textOriginal` exact | 5.922 baris terlibat; **4.771 akan dibuang** | Sumber utama data leakage |
| (c) Near-duplicate (normalisasi dangkal) | 6.368 baris (44,73%) | **Upper bound dangkal — justru kemungkinan underestimate, lihat caveat** |

**Distribusi label pada baris ber-duplikat exact:**

| Label | Jumlah | Porsi |
|-------|--------|-------|
| 1 (judi) | 4.850 | 82% |
| 0 (non-judi) | 1.072 | 18% |

Terdapat **1.151 kelompok teks unik** yang terduplikasi (5.922 − 4.771). Contoh nyata menunjukkan pola spam *template*: satu kalimat promosi disebar puluhan kali dengan merek yang sama.

**Caveat penting pada angka near-duplicate (44,73%):**

Fungsi `normalize_text` saat ini hanya melakukan *lowercase*, hapus URL/mention, hapus tanda baca, dan rapikan spasi. Fungsi ini **tidak menormalkan varian Unicode**. Padahal spam judi (kelas 1) sarat manipulasi Unicode. Akibatnya:

- `𝐏𝐑𝐎𝐁𝐄𝐓855`, `ℙℝ𝕆𝔹𝔼𝕋855`, dan `ＰＲＯＢＥＴ855` diperlakukan sebagai tiga string berbeda meski identik bagi manusia.
- Fakta bahwa near-dup (6.368) hanya **sedikit di atas** exact-dup (5.922) justru **mengonfirmasi** fungsi normalisasi buta terhadap obfuskasi — kalau tidak, near-dup mestinya jauh lebih tinggi.
- **Kesimpulan:** tingkat *template-spam* yang sebenarnya (semantik) **lebih tinggi** dari 44,73%. Angka ini lantai, bukan plafon.

**Implikasi leakage (mengikat desain penelitian):**

- Dedup **harus dilakukan pada data penuh sebelum split**. Jika tidak, teks identik bocor ke train dan test sekaligus → metrik membengkak palsu karena model menghafal.
- Konsekuensi khusus **RM-c**: near-duplicate train↔test membuat FAISS menemukan tetangga nyaris identik, `logit_retrieval` menjadi tinggi palsu, dan keunggulan RM-c jadi artefak — bukan temuan valid. Maka kualitas dedup berdampak langsung ke validitas perbandingan trade-off.

---

## 6. Statistik Panjang Teks (1.4)

![Histogram Panjang Teks](outputs/figures/eda/text_length_histogram.png)

![Boxplot Panjang Teks](outputs/figures/eda/text_length_boxplot.png)

| Metrik token | Kelas 0 | Kelas 1 |
|--------------|---------|---------|
| Median | 9 | 13 |

Komentar **kelas 1 cenderung lebih panjang** — konsisten dengan pola spam: nama merek + pembungkus dekoratif + kalimat pengisi.

**Distribusi panjang token (gabungan) & keputusan `max_length`:**

![Distribusi Panjang Token](outputs/figures/eda/token_length_distribution.png)

| Persentil | Token |
|-----------|-------|
| P95 | 32 |
| P99 | 54 |
| Komentar > 128 token | **0,04%** |

- `do_lower_case` tokenizer = **`True`** → IndoBERT *uncased*. Implikasinya: **case folding manual menjadi redundan** (input untuk keputusan Bagian 3).
- **Keputusan `max_length` = 128** — terkonfirmasi sangat aman. Hanya 0,04% komentar ter-truncate; P99 (54) bahkan jauh di bawah 128. Tidak ada kehilangan informasi berarti, termasuk untuk komentar judi panjang berisi link/kode.

---

## 7. Distribusi Temporal (1.5)

![Distribusi Temporal](outputs/figures/eda/temporal_distribution.png)

- Rentang: **2025-07-31 s/d 2025-10-01** (≈ 62 hari, ~2 bulan).
- Bersifat deskriptif; data terkonsentrasi pada periode pengumpulan komentar dari kanal target.

**Implikasi:** rentang waktu sempit berarti generalisasi temporal di luar periode ini tidak diuji — layak dicatat sebagai batasan (limitation), bukan untuk ditindak di preprocessing.

---

## 8. Audit Kualitas Label (1.6)

Spot-check 15 sampel acak per kelas (seed 42). **Bukan** inter-annotator agreement study (di luar scope S1).

**Karakteristik kelas 1 — taksonomi obfuskasi Unicode:**

| Teknik | Contoh dari data |
|--------|------------------|
| Fullwidth | `ＰＵＬＡＵＷＩＮ`, `Ｇ ＡＲ ＵＤ Ａ Ｈ ０ Ｋ ｌ` |
| Mathematical bold/italic | `𝐏𝐑𝐎𝐁𝐄𝐓 𝟖𝟓𝟓`, `𝑷𝒖𝒍𝒂𝒖𝒘𝒊𝒏`, `𝐃𝐎𝐑𝐀𝟕𝟕` |
| Double-struck | `ℙℝ𝕆𝔹𝔼𝕋855` |
| Small caps | `Pᴜʟᴀᴜᴡɪɴ` |
| Enclosed / negative-squared | `🄿🅄🄻🄰🅄🅆🄸🄽` |
| Variation selector / zero-width | `🅿︎U︎L︎A︎U︎777`, `𝐃⁠​𝐎‍𝐑⁠⁠𝐀⁠‍𝟕‍⁠𝟕` |
| Pembungkus emoji dekoratif | `🤎…🤎`, `🀄…🀄`, `✧…✧`, `🌴 …` |

Merek teridentifikasi: **PULAUWIN, DORA77, PROBET855, PULAU777, GARUDAH0Kl**.

**Kasus ambigu (rawan salah label):**

- `[10809]`, `[7684]` — merek dibingkai sebagai "font desain" dan "mural kampus" (kamuflase konteks).
- `[8684]` — strategi "pura-pura tidak tahu" untuk menyebut merek.
- `[4879]` — merek kurang populer (`ＧＡＲＵＤＡＨ０Ｋｌ`), perlu verifikasi eksternal.

**Anomali pada kelas 0:**

- `[9606]` — *"TOP UP DIAMOND TERBAIK CUMA DI OURASTORE.COM!"* berlabel **0**. Ini promosi top-up game (bukan judi), jadi label 0 secara definisi **benar**, tetapi menandakan ada **komentar promosi non-judi** di kelas 0. Perlu dipastikan definisi label konsisten: target adalah *promosi judi*, bukan *promosi apa pun*.

**Konsistensi keseluruhan:** dari 30 sampel, pelabelan tampak **konsisten** — seluruh sampel kelas 1 memuat nama merek judi. Pasangan `[4082]`/`[5770]` berstruktur sangat mirip → memperkuat sinyal template-spam (lihat §5).

---

## 9. Implikasi Lintas-Pipeline

### 9.1 Imbalance akan memburuk setelah dedup (paling penting)

Karena **82% baris ber-duplikat berlabel 1**, penghapusan duplikat akan memangkas kelas 1 jauh lebih dalam.

> **Estimasi kasar (perlu diverifikasi di Bagian 2):** kelas 1 turun dari 5.677 ke kisaran **~1.700–2.000**; kelas 0 dari 8.560 ke **~7.700**. Rasio berbalik dari **1,5:1 menjadi ≈ 4–4,5:1** (kelas 0 makin dominan).

Konsekuensi:
- Penguatan justifikasi **F1-macro** + **stratified split**; pertimbangan **class weight** pada loss menjadi lebih relevan.
- **RM-c:** vector database FAISS akan didominasi kelas 0; jumlah tetangga kelas 1 mengecil → memengaruhi sensitivitas retrieval terhadap kelas minoritas.
- **Kelayakan data:** ~1.700–2.000 sampel kelas 1, dibagi 70/15/15, menyisakan ~1.200–1.400 untuk train — masih memadai untuk fine-tuning, namun tipis untuk RM-b/RM-c. **Hitung angka pasti di Bagian 2 sebelum melanjutkan.**

### 9.2 Dedup-sebelum-split bukan opsi, melainkan syarat validitas
Sudah ditetapkan di "Prinsip Operasi" `docs/EDA_PLAN.md`; temuan §5 mengonfirmasi besarnya risiko leakage secara empiris.

### 9.3 `max_length = 128` final
Tidak perlu dinaikkan; 0,04% truncation dapat diabaikan.

### 9.4 Case folding kemungkinan redundan
`do_lower_case = True` → IndoBERT menangani lowercasing. Keputusan final di Bagian 3.

### 9.5 Peluang normalisasi Unicode (NFKC) — berdampak besar, biaya rendah
Obfuskasi di sini **dominan berbasis Unicode**, yang sebagian besar dapat dipulihkan dengan satu pemanggilan standar `unicodedata.normalize('NFKC', text)` (mis. `𝐏𝐑𝐎𝐁𝐄𝐓` → `PROBET`, `ＰＵＬＡＵＷＩＮ` → `PULAUWIN`). Ini **berbeda dari** membangun normalizer leetspeak canggih (yang memang future work). NFKC relevan untuk dua hal sekaligus:
- **Near-dup detection** (§5) menjadi jauh lebih akurat.
- **Tokenisasi IndoBERT:** karakter Unicode eksotis kemungkinan menjadi `[UNK]` dan menghapus sinyal merek; NFKC mengembalikannya ke ASCII yang dikenal vocab.

> Rekomendasi: **uji** dampak NFKC di Bagian 2 (terhadap angka near-dup) dan Bagian 3 (terhadap tokenisasi) sebelum memutuskan. Bukan keputusan final di sini.

---

## 10. Pertanyaan Terbuka untuk Bagian 2

Bagian 1 memunculkan urutan pertanyaan yang menjadi tulang punggung Bagian 2:

1. **Apakah NFKC efektif memulihkan obfuskasi Unicode?** — harus dijawab **lebih dulu**, karena fungsi normalisasi adalah fondasi seluruh deteksi near-dup. (terkait §5, §9.5)
2. **Berapa distribusi pasti kelas setelah dedup exact?** — verifikasi estimasi §9.1; tentukan apakah class weight diperlukan dan apakah jumlah kelas 1 masih layak. (terkait §9.1)
3. **Seperti apa distribusi komentar sangat pendek per label?** — `#STOPJUDI`/`#STOPJUDOL` (valid, kelas 0) vs `.`/`P` (noise) menunjukkan threshold panjang saja tidak cukup; perlu kriteria gabungan. (terkait §8)
4. **Berapa proporsi komentar non-Indonesia, dan berlabel apa?** — belum diukur di Bagian 1.
5. **Apakah outlier panjang ekstrem berupa noise atau konten valid?** — tinjau contoh sebelum memutuskan.

---

## 11. Catatan Reproduktibilitas & Teknis

- **Random seed:** 42 (sampling manual). Konsisten dengan rencana split.
- **Figur tersimpan** di `outputs/figures/eda/`: `label_distribution.png`, `text_length_histogram.png`, `text_length_boxplot.png`, `token_length_distribution.png`, `temporal_distribution.png`.
- **Catatan tokenizer:** notebook kini **diselaraskan ke `indobenchmark/indobert-base-p2`** (sel 1.4 dan seluruh pengukuran token). Sebelumnya Bagian 1 dijalankan dengan p1; karena kedua model **berbagi vocabulary WordPiece yang sama** (`do_lower_case=True`), statistik panjang token pada laporan ini **tetap valid**. Penyelarasan ke p2 penting terutama untuk Bagian 2 (pengukuran presisi `[UNK]`). ✅ *(sudah dieksekusi)*
- **Lingkungan:** EDA dijalankan lokal (pandas/CPU), tanpa GPU — sesuai catatan environment `docs/EDA_PLAN.md`.

---

---

## 12. Susulan — Pengukuran Deskriptif Tambahan

Lima analisis susulan (⏳ pada `docs/EDA_PLAN.md`) yang murni komputasi dari data mentah, melengkapi Bagian 1 sebelum lanjut ke Bagian 2. Semua bersifat **deskriptif** — tidak ada baris yang diubah/dibuang.

> Basis hitung: `valid` = data non-missing = **14.227** baris (raw kelas 0/1 = 8.550 / 5.677; catatan: 10 baris `textOriginal` kosong seluruhnya berlabel 0, sehingga L0 valid 8.550 vs 8.560 di §1).

### 12.1 Distribusi Kelas Pasti Pasca-Dedup Exact

| Metrik | Nilai |
|--------|-------|
| Baris setelah dedup exact `textOriginal` | 9.465 |
| Kelas 0 | 7.702 (81,37%) |
| Kelas 1 | 1.763 (18,63%) |
| **Rasio** | **4,37 : 1** (kelas 0 dominan) |
| Kelas 1: perubahan | 5.677 → 1.763 (**turun 3.914**) |
| Kelas 0: perubahan | 8.550 → 7.702 (turun 848) |
| Kelayakan kelas 1 (split 70/15/15) | train ~1.234 · val ~264 · test ~264 |

**Estimasi §9.1 terkonfirmasi.** Rasio membalik dari 1,51:1 (mentah) menjadi **4,37:1** — dedup memangkas kelas 1 jauh lebih dalam (82% duplikat memang berlabel 1). Menguatkan **F1-macro + stratified split + class weight** (final di Bagian 3). Kelas 1 menyisakan ~1.234 sampel train — memadai untuk RM-a, tipis-tapi-cukup untuk RM-b/RM-c.

### 12.2 Grup Duplikat Exact Ber-konflik Label

| Metrik | Nilai |
|--------|-------|
| Grup teks identik dengan >1 label | **3 grup** |
| Baris terdampak | **32 baris** |

Ketiga grup adalah varian brand *"Garuda-Hoki"* (mis. `Aku pertama depo jp diGaruda-Ho ki😚` — 17×, label {1:16, 0:1}). Di tiap grup, **mayoritas berlabel 1** dengan 1 baris label 0 menyimpang (kemungkinan salah-label anotator). Skala sangat kecil (0,2%), dan aturan **assign label 1** (dikunci Bagian 3) sejalan dengan mayoritas.

### 12.3 Komentar Sangat Pendek per Label

| Threshold | n | L0 | L1 |
|-----------|-----|-----|-----|
| ≤ 1 karakter | 184 | 184 | **0** |
| ≤ 5 karakter | 809 | 809 | **0** |
| ≤ 10 karakter | 1.587 | 1.587 | **0** |
| ≤ 1 kata | 1.256 | 1.256 | **0** |
| ≤ 2 kata | 2.262 | 2.252 | 10 |
| ≤ 5 kata | 5.370 | 4.659 | 711 |

- Komentar ≤2 kata kelas 0: 2.252 (mengandung alfanumerik 1.894; emoji/simbol murni **358**).
- Kampanye `stopjud*`: **5 komentar** (semua L0), 4 di antaranya ≤2 kata.

**Seluruh komentar sangat pendek adalah kelas 0** — komentar judi praktis tidak pernah sangat pendek (butuh ruang menampilkan brand). Threshold panjang-saja hanya akan membuang kelas 0 dan berisiko menghapus kampanye valid → keputusan filtering pendek harus hati-hati (Bagian 3).

### 12.4 Proporsi Komentar Non-Indonesia

| Metrik | Nilai |
|--------|-------|
| Non-Indonesia (termasuk `unknown`) | 3.750 (26,36%) |
| Non-Indonesia (bahasa terdeteksi, excl. unknown) | 3.295 (23,16%) |
| `unknown` (gagal deteksi) | 455 |
| Label non-ID (excl. unknown) | L0=2.864 · L1=431 |

Top bahasa non-ID: `tl` (Tagalog) 733, `en` 304, `sw` 271, `de` 243, `so` 225, `fi` 163. Angka 23% ini **menyesatkan/over-estimate**: 733 baris "Tagalog" dan ratusan "sw/de/so/fi" hampir seluruhnya **teks Indonesia informal/pendek yang salah-deteksi** (langdetect lemah pada slang & teks pendek). Keputusan Bagian 3: **pertahankan** (drop berbasis bahasa tak reliabel). **Batasan scope:** deteksi ringan, bukan language-model berat (future work bila perlu).

### 12.5 Sampel Outlier Panjang Ekstrem

| Persentil | Threshold | Baris | (di antaranya L1) |
|-----------|-----------|-------|-------------------|
| P95 | > 115 char | 699 | 228 |
| P99 | > 240 char | 142 | 22 |
| P99,9 | > 404 char | 15 | 0 |

Top-20 komentar terpanjang seluruhnya **kelas 0** berupa diskusi esports (MLBB) yang koheren (mis. `[1381]` 886 char / 143 kata). Tidak ada pola spam-pengulangan di ekor atas. **Outlier = konten valid, bukan noise.** Keputusan Bagian 3: **pertahankan** (`max_length=128` token sudah menangani truncation).

### Ringkasan Susulan

| # | Analisis | Angka final |
|---|----------|-------------|
| 12.1 | Distribusi pasca-dedup exact | L0=7.702 / L1=1.763 — **rasio 4,37:1** |
| 12.2 | Grup duplikat konflik label | 3 grup / 32 baris (brand "Garuda-Hoki", mayoritas L1) |
| 12.3 | Komentar sangat pendek | ≤5 char: 809 (semua L0); kampanye stopjud* = 5 |
| 12.4 | Non-Indonesia | 23,16% (over-estimate; false-positive langdetect) |
| 12.5 | Outlier panjang | konten valid (diskusi esports L0), bukan noise |

---

*Laporan ini menjadi input untuk Bagian 2 (`docs/EDA_REPORT_BAGIAN2.md`) & Bagian 3 (`docs/EDA_REPORT_BAGIAN3.md`), dan bahan Bab 4 (Hasil & Pembahasan).*
