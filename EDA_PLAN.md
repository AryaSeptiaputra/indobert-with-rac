# EDA Plan — IndoBERT-with-RAC

Dokumen ini mendefinisikan struktur dan tujuan setiap bagian EDA sebelum implementasi ke `notebooks/01_eda.ipynb`. EDA hanya **mengukur dan memutuskan**; eksekusi dedup, penyaringan, transformasi, dan split dilakukan di fase preprocessing terpisah (`02_preprocessing.ipynb`).

> **Catatan revisi (2026-07-06):** Struktur di bawah menggantikan versi sebelumnya. Penomoran Bagian 1/2/3 dipertahankan agar konsisten dengan `EDA_REPORT_BAGIAN1.md` dan notebook yang sudah ada, tapi definisi tiap Bagian diubah. Sebelumnya Bagian 2 dan 3 dipisah berdasarkan *jenis keputusan* (filtering vs transformasi) — ini membuat pengukuran baru (mis. uji efektivitas NFKC) "nyempil" di dalam bagian yang seharusnya berisi keputusan, dan menimbulkan kontradiksi antara diagram linear dengan catatan "sifat iteratif". Sekarang pemisahan berdasarkan *jenis aktivitas*: deskripsi data mentah → uji dampak kandidat transformasi → keputusan. Total pekerjaan tidak bertambah, hanya pengelompokannya yang diperbaiki.

---

## Prinsip Operasi (Tulang Punggung)

Urutan operasi yang mengikat seluruh pipeline, dari data mentah sampai siap latih:

```
inspeksi (tanpa mengubah data)
  → deduplikasi pada data penuh
    → penyaringan baris lain
      → split stratified 70:15:15
        → transformasi teks
          → [khusus RM-c] pembangunan index FAISS hanya dari train set
```

Dua aturan non-negotiable:

1. **Deduplikasi sebelum split.** Spam promosi judi hampir selalu copy-paste, sehingga `textOriginal` yang identik atau nyaris identik tersebar di banyak komentar. Jika split dilakukan sebelum dedup, teks yang sama bisa bocor ke train dan test sekaligus (data leakage) — metrik terlihat bagus padahal model hanya menghafal.

2. **Index FAISS hanya dari train set.** Ekstra penting untuk RM-c. Jika ada near-duplicate train↔test, retrieval FAISS akan menemukan tetangga nyaris identik dari train dan `logit_retrieval` menjadi tinggi palsu — RM-c terlihat unggul secara tidak fair dibanding RM-a/RM-b, dan temuan trade-off jadi tidak valid.

> Urutan ini ditetapkan dari desain penelitian, bukan dari hasil EDA — jadi sudah berlaku sejak sekarang, terlepas dari apa yang nanti ditemukan di data.

---

## Bagian 1 — Pengukuran Deskriptif Data Mentah

**Status: sebagian sudah dieksekusi.** Lihat `EDA_REPORT_BAGIAN1.md` untuk hasil lengkap dari analisis di bawah yang ditandai ✅. Item yang ditandai ⏳ adalah **susulan** yang perlu dilengkapi sebelum lanjut ke Bagian 2 — semuanya murni komputasi dari data mentah, tidak butuh keputusan transformasi apa pun lebih dulu.

**Tujuan:** Memotret bentuk dan karakteristik dasar dataset apa adanya, tanpa mengubah atau memutuskan apa pun.

**Sudah dikerjakan (✅):**

| Analisis | Detail |
|----------|--------|
| Shape & kolom | Jumlah baris, kolom relevan (`textOriginal`, label) vs metadata |
| Distribusi label | Proporsi kelas 0 vs 1 (raw) + visualisasi |
| Missing values | NaN pada `textOriginal` dan `label` |
| Duplikat exact & near-dup dangkal | `commentId` (artefak API) vs `textOriginal` (sumber leakage); estimasi near-dup dangkal (dengan caveat buta-Unicode) |
| Statistik panjang teks | Distribusi karakter & token per kelas, persentil, keputusan `max_length` |
| Distribusi temporal | Sebaran `publishedAt` |
| Sampel inspeksi & audit label | ~20–30 sampel per kelas, taksonomi obfuskasi Unicode |

**Susulan yang masih perlu dijalankan (⏳):**

| Analisis | Kenapa ini murni Bagian 1, bukan Bagian 2/3 |
|----------|----------------------------------------------|
| Distribusi kelas **pasti** pasca-dedup exact (`drop_duplicates(subset='textOriginal').label.value_counts()`) | Langsung dihitung dari data yang sudah ada (duplikat sudah teridentifikasi) — tidak menunggu keputusan apa pun |
| Cek grup duplikat exact dengan **konflik label** (teks sama, label beda) — hitung berapa banyak grup & baris terdampak | Deteksi, bukan keputusan. Aturan penanganannya sudah dikunci di Bagian 3 (lihat prinsip tetap), tapi *jumlahnya* perlu diketahui di sini |
| Distribusi panjang komentar sangat pendek per label (mis. <5 karakter) | Sudah ada data panjang token; tinggal cross-tab dengan label untuk menyusun kriteria "terlalu pendek" |
| Proporsi komentar non-Indonesia | Deteksi bahasa langsung dari teks mentah, tidak butuh transformasi apa pun |
| Sampel outlier panjang ekstrem (mis. top-20 terpanjang) untuk ditinjau manual | Sama sifatnya dengan audit label yang sudah dikerjakan — observasi manual, bukan transformasi |

**Catatan implikasi (jangan berhenti di angka):**

- Imbalance raw (1,51:1) menyesatkan jika dibaca tanpa konteks dedup — item susulan pertama di atas mengonfirmasi angka pasti pasca-dedup, jadi jangan lanjut ke Bagian 2 sebelum ini selesai.
- `max_length = 128` sudah terkonfirmasi aman (final, lihat Bagian 3).

**Output yang diharapkan:** semua angka final/eksak (bukan estimasi) yang jadi basis pengujian di Bagian 2 dan keputusan di Bagian 3, plot pendukung, dan catatan kriteria label. Tidak ada satu baris pun yang diubah, dibuang, atau disimpan di Bagian 1.

**Batasan scope:** berhenti di spot-check label dan deteksi bahasa sederhana. **Jangan** lakukan inter-annotator agreement study atau language-detection model yang berat — di luar scope, catat sebagai future work bila perlu.

---

## Bagian 2 — Uji Dampak Kandidat Transformasi

**Status: belum dikerjakan** — menunggu Bagian 1 (termasuk susulan) selesai penuh.

**Tujuan:** Mengukur *efek* dari menerapkan sebuah kandidat transformasi secara sementara (tidak dipersist), sebelum ada keputusan pakai/tidak. Ini beda sifat dari Bagian 1: di Bagian 1 kita mendeskripsikan data apa adanya, di sini kita **mencoba** sesuatu pada data (di memory/temporary variable, bukan menimpa file) lalu mengukur akibatnya.

| Kandidat transformasi | Cara uji | Metrik efek yang diukur |
|---|---|---|
| **NFKC (`unicodedata.normalize('NFKC', text)`)** | Terapkan sementara ke seluruh `textOriginal` | (a) perubahan jumlah near-duplicate terdeteksi sebelum vs sesudah; (b) perubahan jumlah token `[UNK]` saat tokenisasi, khususnya pada sampel kelas 1 yang sarat obfuskasi Unicode (`𝐏𝐑𝐎𝐁𝐄𝐓`, `ＰＵＬＡＵＷＩＮ`, dll — lihat §8 laporan Bagian 1) |
| Emoji removal | Hapus emoji dari seluruh teks secara sementara | Berapa baris jadi string kosong / hanya whitespace setelahnya — angka ini yang menentukan apakah Bagian 3 perlu aturan drop tambahan |
| Case folding | — | **Tidak perlu diuji.** `do_lower_case=True` sudah dikonfirmasi di Bagian 1 (§9.4) — lowercase manual sudah pasti redundan. Cukup dicatat sebagai "dilewati" di Bagian 3 |

**Prasyarat teknis:** jalankan uji tokenisasi (kolom NFKC) dengan **`indobenchmark/indobert-base-p2`**, bukan p1 yang dipakai di Bagian 1 lama — ini koreksi yang sudah disepakati, dan penting di sini karena kita mengukur perubahan rate `[UNK]` secara presisi, bukan sekadar statistik panjang token (yang memang sudah valid lintas p1/p2 karena vocab sama).

**Output yang diharapkan:** tabel before/after angka efek per kandidat transformasi. **Belum ada keputusan pakai/tidak di sini** — itu baru terjadi di Bagian 3.

---

## Bagian 3 — Keputusan Terpadu (Filtering + Transformasi)

**Status: belum dikerjakan** — menunggu Bagian 1 & 2 selesai.

**Tujuan:** Mengambil seluruh keputusan penyaringan baris dan transformasi teks berdasarkan angka dari Bagian 1 & 2. Filtering dan transformasi digabung jadi satu bagian karena keduanya memang saling memengaruhi (bukan dua tahap linear terpisah) — contoh konkret: kalau emoji dihapus (keputusan transformasi) dan itu menyisakan baris kosong, itu memaksa keputusan filtering baru. Perlakukan sebagai **loop**, bukan garis lurus, dan itu berlaku sebagai struktur bagian ini — bukan disclaimer tempelan seperti versi sebelumnya.

**A. Prinsip tetap (tidak bergantung data, dari desain penelitian):**

- Dedup dijalankan pada data penuh sebelum split.
- Exact duplicate pada `textOriginal` dibuang.
- Missing pada `textOriginal` atau label → drop.
- Konflik label pada grup duplikat exact (teks sama, label beda) → default **assign label 1**.
- URL / mention / angka → ganti placeholder (`[URL]`, dll), **bukan** dihapus total — ini sinyal kuat promosi judi.
- Obfuskasi residual (`sl0t`, `g4cor`, `j*di`) setelah NFKC → dokumentasikan saja, **jangan** bangun normalizer leetspeak canggih (robustness obfuskasi adalah future work, bukan yang diuji).
- Tokenisasi pakai WordPiece bawaan IndoBERT **p2**, bukan tokenisasi manual.

**B. Sudah dikunci dari hasil Bagian 1 (data-dependent, sudah final):**

- Drop 10 baris `textOriginal` kosong.
- `max_length = 128` (P99 = 54 token, truncation hanya 0,04%).
- Case folding manual dilewati (`do_lower_case=True`).
- Stratified split 70/15/15.

**C. Menunggu hasil Bagian 1 susulan & Bagian 2 (masih terbuka):**

- NFKC dipakai atau tidak — dari hasil uji Bagian 2.
- Threshold "komentar terlalu pendek" — dari distribusi Bagian 1 susulan.
- Emoji dihapus/dikonversi jadi token, dan cara menangani baris yang jadi kosong akibatnya — loop balik ke filtering.
- Komentar non-Indonesia: dibuang atau dipertahankan — dari proporsi Bagian 1 susulan.
- Outlier panjang ekstrem: dibiarkan atau dibuang — dari sampel Bagian 1 susulan.
- Normalisasi slang/singkatan ringan: cakupan kamus — dari severitas di Bagian 1.
- Class weight pada loss: perlu atau tidak — dari distribusi kelas pasti pasca-dedup (Bagian 1 susulan), sekaligus **feasibility gate** untuk RM-b/RM-c.

**Output wajib:** tabel before/after count baris (data awal → dibuang karena duplikat sekian → missing sekian → … → data final) dan daftar keputusan transformasi final. Langsung dipakai sebagai input `02_preprocessing.ipynb` dan bahan Bab 4.

---

## Artefak Output Fase EDA

- `notebooks/01_eda.ipynb` berisi tiga section (Bagian 1, 2, 3) sesuai struktur di atas.
- Tabel distribusi kelas (raw & pasca-dedup) & statistik panjang teks.
- Tabel before/after angka efek kandidat transformasi (Bagian 2).
- Tabel before/after count penyaringan baris (Bagian 3).
- Catatan kriteria label + daftar keputusan preprocessing final (input fase berikutnya sekaligus bahan Bab 4).
- Random seed yang didokumentasikan (sampling manual & split) — reproducibility.

---

## Urutan Eksekusi di Notebook

```
Bagian 1 (Deskriptif — lengkapi item susulan ⏳ dulu)
  → Bagian 2 (Uji dampak kandidat transformasi — perlu angka Bagian 1 lengkap)
    → Bagian 3 (Keputusan terpadu, loop filtering↔transformasi internal — perlu Bagian 1 & 2)
```

Arah makro (1→2→3) tetap linear: setiap bagian butuh output bagian sebelumnya. Yang berbeda dari versi lama: loop hanya terjadi **di dalam** Bagian 3 (antara sub-keputusan filtering dan transformasi), bukan lagi menyilang Bagian 2↔3 sebagai dua bagian terpisah — jadi tidak ada lagi kontradiksi antara diagram dan catatan iteratif.

---

## Catatan Environment

EDA tidak butuh GPU — ini murni kerja pandas. **Jalankan di lokal** (Jupyter di laptop). Simpan kredit Vast.ai untuk fase training (RM-a/b/c); menyewa GPU hanya untuk `describe()` dan `value_counts()` adalah pemborosan budget.