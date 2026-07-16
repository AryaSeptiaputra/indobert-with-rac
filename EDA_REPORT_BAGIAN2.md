# Laporan EDA — Bagian 2: Uji Dampak Kandidat Transformasi

| | |
|---|---|
| **Proyek** | Analisis Trade-off Performa & Efisiensi Adaptasi IndoBERT dengan RAC untuk Deteksi Komentar Promosi Judi Daring |
| **Penulis** | Arya Eka Septiaputra (152022190) — Informatika ITENAS 2026 |
| **Sumber data** | `dataset/raw/data_labeling.csv` |
| **Random seed** | 42 |
| **Notebook** | `notebooks/01_eda.ipynb` (Bagian 2) |
| **Tokenizer** | `indobenchmark/indobert-base-p2` (`do_lower_case=True`) |
| **Sifat fase** | Uji dampak — **mengukur efek**, belum ada keputusan pakai/tidak |

> Bagian 2 mengukur *efek* menerapkan sebuah kandidat transformasi **secara sementara** (in-memory, tidak dipersist), sebelum ada keputusan. Ini beda sifat dari Bagian 1 (mendeskripsikan data apa adanya): di sini kita **mencoba** sesuatu lalu mengukur akibatnya. **Seluruh keputusan pakai/tidak ada di Bagian 3.**

---

## 1. Ringkasan Eksekutif

Tiga kandidat transformasi diuji. **NFKC** adalah temuan terpenting: dampaknya kecil pada jumlah duplikat, tetapi **besar** pada kualitas tokenisasi kelas 1 (judi).

| Kandidat | Metrik efek | Before | After | Efek |
|---|---|---|---|---|
| **NFKC** | Near-duplicate terbuang (kunci NFKC vs exact) | 4.762 | 4.815 | **+53** varian Unicode |
| **NFKC** | Baris **kelas 1** dengan ≥1 `[UNK]` | 90,51% | 53,60% | **−36,9 pp** |
| **NFKC** | Total token `[UNK]` kelas 1 | 9.160 | 5.489 | **−40,1%** |
| **NFKC** | Total token `[UNK]` (semua kelas) | 12.746 | 8.920 | −30,0% |
| **Emoji removal** | Baris jadi string kosong | 0 | 353 (semua L0) | butuh drop bila diterapkan |
| **Case folding** | — | — | — | **dilewati** (redundan) |

**Kesimpulan pengukuran (bukan keputusan):**
1. NFKC memulihkan mayoritas obfuskasi Unicode kelas 1 ke ASCII yang dikenal vocab IndoBERT — **bukti terkuat** untuk mengadopsinya sebagai fondasi normalisasi.
2. Emoji removal *sebagai transformasi wajib* justru menciptakan 353 baris kosong (semuanya kelas 0) — akan memaksa aturan drop baru.
3. Case folding manual dipastikan redundan (`do_lower_case=True`).

---

## 2. NFKC — Efek pada Deteksi Near-Duplicate (2.1)

`normalize_nfkc` membuang karakter tak-terlihat (ZWSP/ZWNJ/ZWJ/variation selector) lalu `unicodedata.normalize('NFKC', …)`.

**Duplikat terbuang & distribusi kelas pasca-dedup, per definisi kunci** (dari 14.227 baris valid):

| Kunci | Terbuang | % baris dup | Total tersisa | L0 | L1 | Rasio |
|-------|----------|-------------|---------------|-----|-----|-------|
| Mentah (exact) | 4.762 | 41,55% | 9.465 | 7.702 | 1.763 | 4,37:1 |
| Dangkal (lama) | 5.193 | 44,69% | 9.034 | 7.335 | 1.699 | 4,32:1 |
| **NFKC saja** | **4.815** | 41,84% | 9.412 | 7.702 | **1.710** | **4,50:1** |
| NFKC + dangkal | 5.253 | 44,84% | 8.974 | 7.335 | 1.639 | 4,48:1 |

**Uji recovery pada sampel obfuskasi nyata (audit §8 laporan Bagian 1):**

| Input | NFKC output | Status |
|-------|-------------|--------|
| `ＰＵＬＡＵＷＩＮ` (fullwidth) | `PULAUWIN` | ✅ pulih |
| `ℙℝ𝕆𝔹𝔼𝕋855` (double-struck) | `PROBET855` | ✅ pulih |
| `🄿🅄🄻🄰🅄🅆🄸🄽` (enclosed) | `PULAUWIN` | ✅ pulih |
| `Pᴜʟᴀᴜᴡɪɴ` (small caps) | `Pᴜʟᴀᴜᴡɪɴ` | ❌ bertahan |
| `🤎𝐏𝐑𝐎𝐁𝐄𝐓 𝟖𝟓𝟓🤎` (math bold + emoji) | `🤎PROBET 855🤎` | ⚠️ huruf pulih, emoji tetap |

**Temuan:**
- NFKC menambah deteksi duplikat **modest tapi nyata**: `NFKC saja` membuang **+53 baris** dibanding `exact` (4.815 vs 4.762). Tambahan ini murni varian Unicode kelas 1 — dua komentar yang identik bagi manusia (`𝐏𝐑𝐎𝐁𝐄𝐓855` ≡ `ℙℝ𝕆𝔹𝔼𝕋855`) yang sebelumnya lolos sebagai "beda byte".
- Ini **mengonfirmasi caveat §5 laporan Bagian 1**: angka near-dup dangkal (44,73%) adalah *lantai*, bukan plafon — normalisasi buta-Unicode memang melewatkan template-spam ber-obfuskasi.
- Efek pada **distribusi kelas**: `NFKC saja` menurunkan kelas 1 dari 1.763 → 1.710 (−53), menaikkan rasio ke **4,50:1**. Kelas 0 tak berubah (7.702) — konsisten dengan fakta obfuskasi terkonsentrasi di kelas 1.
- **Relevansi RM-c:** +53 near-duplicate Unicode ini justru vektor leakage yang paling berbahaya untuk FAISS (tetangga nyaris identik train↔test). Menangkapnya memperkuat validitas perbandingan trade-off.

---

## 3. NFKC — Efek pada Tokenisasi `[UNK]` (2.2) — *Pengukuran Kunci*

Dihitung dengan tokenizer **`indobert-base-p2`** (koreksi dari p1; presisi `[UNK]` butuh model final). `[UNK]` = token yang tak dikenal vocab WordPiece → sinyal hilang.

| Scope | n | `[UNK]` sebelum | %baris | `[UNK]` sesudah | %baris | `[UNK]` turun |
|-------|-----|-----------------|--------|-----------------|--------|---------------|
| **ALL** | 14.227 | 12.746 | 58,75% | 8.920 | 43,66% | −3.826 (−30,0%) |
| **L1 (judi)** | 5.677 | 9.160 | **90,51%** | 5.489 | **53,60%** | −3.671 (**−40,1%**) |
| **L0 (non-judi)** | 8.550 | 3.586 | 37,66% | 3.431 | 37,06% | −155 (−4,3%) |

**Contoh token brand yang pulih setelah NFKC:**

| Input | Tokenisasi sebelum | Tokenisasi sesudah |
|-------|--------------------|--------------------|
| `ＰＵＬＡＵＷＩＮ` | `['[UNK]']` | `['pulau', '##win']` |
| `ℙℝ𝕆𝔹𝔼𝕋855` | `['[UNK]']` | `['prob', '##et', '##85', '##5']` |
| `🄿🅄🄻🄰🅄🅆🄸🄽` | `['[UNK]']` | `['pulau', '##win']` |
| `Pᴜʟᴀᴜᴡɪɴ` | `['[UNK]']` | `['[UNK]']` (small caps tak terpulihkan) |

**Temuan (decisive):**
- Dampak NFKC **besar dan terkonsentrasi di kelas 1**. Sebelum NFKC, **90,51% komentar judi** mengandung setidaknya satu `[UNK]` — artinya nama brand (sinyal utama) lenyap menjadi token buta pada 9 dari 10 komentar judi. Setelah NFKC, turun ke **53,60%**, dan total `[UNK]` kelas 1 turun **40,1%**.
- Kelas 0 nyaris tak terpengaruh (−4,3%) — wajar, obfuskasi Unicode adalah ciri khas spam judi.
- Sisa `[UNK]` pasca-NFKC berasal dari (a) **small caps** (tak punya decomposition Unicode) dan (b) brand yang **lengket dengan emoji** tanpa spasi (`🤎𝐏𝐑𝐎𝐁𝐄𝐓` → `🤎PROBET` masih satu token asing). Penanganan lanjutan (normalizer leetspeak/small-caps) sengaja **tidak** dibangun — di luar scope (future work).
- Inilah **bukti empiris terkuat** untuk mengadopsi NFKC sebagai langkah fondasi (diputus final di Bagian 3).

---

## 4. Emoji Removal — Efek pada Baris Kosong (2.3)

Emoji dihapus sementara memakai **regex rentang-Unicode** (blok pictographs, dingbats, regional indicators, arrows, variation selectors, mahjong/cards). Library `emoji` sengaja **tidak** dipakai agar notebook bebas dependensi tambahan.

| Metrik | Nilai |
|--------|-------|
| Baris kosong **sebelum** emoji removal | 0 |
| Baris kosong **sesudah** emoji removal | 353 |
| Baris **baru** jadi kosong akibat emoji removal | **353** (L0=353, L1=0) |

Contoh baris yang jadi kosong: `😂😂😂😂😂😂`, `🎉🎉🎉🎉`, `🙋`, `☝️😁` — seluruhnya reaksi emoji-murni.

**Temuan:**
- Menghapus emoji membuat **353 baris menjadi kosong, semuanya kelas 0** (reaksi non-judi). **Tidak ada satu pun kelas 1** yang bergantung sepenuhnya pada emoji — brand judi selalu memuat teks/angka.
- Artinya: emoji removal *sebagai transformasi wajib* justru **menciptakan pekerjaan filtering baru** (353 baris kosong harus di-drop) tanpa manfaat untuk kelas target. Ini contoh konkret loop filtering↔transformasi yang dibahas di Bagian 3.

---

## 5. Case Folding — Dilewati (2.4)

**Tidak diuji.** `tokenizer.do_lower_case = True` pada `indobert-base-p2` → IndoBERT sudah *uncased* dan menangani lowercasing secara internal. Lowercase manual dipastikan **redundan**. Dicatat "dilewati" di Bagian 3 tanpa pengukuran (sesuai `EDA_PLAN.md`).

---

## 6. Catatan Reproduktibilitas & Teknis

- **Tokenizer:** seluruh pengukuran `[UNK]` memakai `indobenchmark/indobert-base-p2` (bukan p1 seperti Bagian 1 lama) — koreksi yang disepakati. Statistik panjang token Bagian 1 tetap valid lintas p1/p2 (vocab WordPiece sama).
- **Random seed:** 42.
- **Lingkungan:** lokal (pandas/CPU + tokenizer transformers), tanpa GPU — sesuai catatan environment `EDA_PLAN.md`.
- **Sifat data:** tidak ada baris yang dipersist; seluruh transformasi hanya di memory untuk pengukuran.

---

*Laporan ini menjadi input keputusan **Bagian 3** (khususnya keputusan NFKC & emoji) dan bahan Bab 4 (Hasil & Pembahasan).*
