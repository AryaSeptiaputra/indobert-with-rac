# DATASET.md — Dokumentasi Dataset & Preprocessing

Deteksi komentar promosi judi daring berbahasa Indonesia di YouTube (klasifikasi biner: `0` = non-judi, `1` = promosi judi). Dokumen ini merangkum dataset final dan pipeline preprocessing yang menghasilkannya. Keputusan lengkap ada di `EDA_REPORT_BAGIAN1.md`, `EDA_REPORT_BAGIAN2.md`, dan `EDA_REPORT_BAGIAN3.md`.

## Sumber

| | |
|---|---|
| File mentah | `dataset/raw/data_labeling.csv` (14.237 baris, komentar YouTube berlabel) |
| Kolom dipakai | `textOriginal` (teks), `label` (0/1) |
| Kolom lain | metadata (commentId, author, timestamp, dll) — di-drop |
| Rentang waktu | 2025-07-31 s/d 2025-10-01 (~62 hari) |
| Random seed | 42 |

## Pipeline Preprocessing

Dieksekusi oleh `notebooks/02_preprocessing.ipynb` (fungsi di `src/preprocessing.py`).

```
load raw (textOriginal, label)
 → drop missing
 → nfkc_key = normalize_nfkc(textOriginal)
 → resolve konflik label (grup nfkc_key >1 label → assign 1)
 → dedup by nfkc_key (keep first)          → 9.412 baris  (reproduksi EDA)
 → stratified split 70/15/15 (seed 42)
 → clean_text(textOriginal) per subset     (transformasi setelah split)
 → guard anti-leakage (buang text_clean duplikat lintas-split dari val/test)
 → simpan splits + processed + metadata
```

### Prinsip kunci

- **Dedup sebelum split** pada data penuh (anti-leakage; kritis untuk RM-c/FAISS). Kunci = **NFKC-exact**, menangkap obfuskasi Unicode kelas 1 (fullwidth/double-struck/enclosed).
- **NFKC** = fondasi normalisasi. Bukti EDA: baris kelas 1 ber-`[UNK]` turun 90,5% → 53,6% (Bagian 2).
- **Konflik label** grup duplikat → **assign 1** (3 grup / 32 baris; mayoritas memang label 1).
- **Guard anti-leakage** kedua: placeholder dapat mengolapskan template-spam beda-URL/angka menjadi `text_clean` identik → duplikat lintas-split dibuang dari val/test (train dipertahankan).

### Transformasi teks (`clean_text`)

| Transform | Aturan | Contoh |
|-----------|--------|--------|
| NFKC | pulihkan varian Unicode | `𝐏𝐑𝐎𝐁𝐄𝐓` → `PROBET`, `ＰＵＬＡＵＷＩＮ` → `PULAUWIN` |
| URL → `[URL]` | `https?://…`, `www.…`, `t.me/…` | `https://x.com` → `[URL]` |
| Mention → `[MENTION]` | `@\w+` | `@andiw_yt` → `[MENTION]` |
| Angka → `[NUM]` | **hanya digit berdiri sendiri** (`\b\d+\b`) | `depo 50000` → `depo [NUM]` |
| — dipertahankan | brand alfanumerik | `DORA77`, `PROBET855` tetap utuh |

**Tidak** dilakukan: lowercase manual (`do_lower_case=True`), penghapusan emoji, penghapusan tanda baca, drop komentar pendek/non-Indonesia/outlier panjang (lihat justifikasi `EDA_REPORT_BAGIAN3.md`).

> Catatan perilaku: brand yang menuliskan angka **terpisah spasi** (mis. `PROBET 855`) akan menjadi `PROBET [NUM]` karena `855` menjadi token digit berdiri sendiri. Huruf brand (sinyal utama) tetap dipertahankan.

## Dataset Final

### Jumlah baris

| Tahap | Baris |
|-------|-------|
| Data awal | 14.237 |
| − Missing `textOriginal`/`label` | −10 → 14.227 |
| − Duplikat (NFKC-exact) | −4.815 → 9.412 |
| − Guard leakage (text_clean lintas-split) | −17 → **9.395** |

### Distribusi per split (stratified 70/15/15)

| Split | n | L0 | L1 | L1 % |
|-------|-----|-----|-----|------|
| train | 6.588 | 5.391 | 1.197 | 18,17% |
| val | 1.402 | 1.145 | 257 | 18,33% |
| test | 1.405 | 1.149 | 256 | 18,22% |
| **total** | **9.395** | 7.685 | 1.710 | 18,20% |

Rasio imbalance ≈ **4,5 : 1** (kelas 0 dominan). Metrik utama: **F1-macro** (bukan accuracy).

### Class weights (dari train, sklearn `balanced`)

```json
{ "0": 0.6110, "1": 2.7519 }
```

Dipakai pada loss berbobot (mis. `CrossEntropyLoss(weight=...)`) untuk menangani imbalance — bukan dengan membuang baris.

## Artefak Output

| Path | Isi |
|------|-----|
| `dataset/splits/train.csv` · `val.csv` · `test.csv` | kolom `textOriginal, text_clean, label` |
| `dataset/processed/data_clean.csv` | gabungan ketiga split + kolom `split` |
| `dataset/processed/metadata.json` | count before/after, distribusi, class weights, config |

Skema kolom split: `textOriginal` (mentah, untuk traceability), `text_clean` (input model), `label` (0/1).

## Tokenisasi (untuk fase modeling)

Dikunci di `src/dataset.py`:

- Base tokenizer: **`indobenchmark/indobert-base-p2`** (`do_lower_case=True`).
- **max_length = 128** (P99 panjang token `text_clean` = 58; hanya 0,08% > 128).
- Placeholder `[URL]`/`[MENTION]`/`[NUM]` didaftarkan sebagai **additional_special_tokens** → tiap placeholder 1 token utuh. Ukuran vocab: 30.521 → **30.524**.

> ⚠️ **Wajib di semua model (RM-a/b/c):** setelah memuat bobot pra-latih, panggil
> `model.resize_token_embeddings(len(tokenizer))` sebelum training/inference.

## Reproduktibilitas

- Seed **42** di seluruh langkah (dedup deterministik, split stratified).
- Jalankan ulang: `notebooks/02_preprocessing.ipynb` (butuh `dataset/raw/data_labeling.csv`).
- Modul: `src/preprocessing.py` (pure pandas/sklearn), `src/dataset.py` (torch/transformers).

## Catatan untuk RM-c (RAC)

Index FAISS **hanya dibangun dari train set** (`dataset/splits/train.csv`). Membangunnya dari data yang memuat val/test akan menyebabkan retrieval menemukan tetangga nyaris identik → `logit_retrieval` tinggi palsu dan keunggulan RM-c menjadi artefak (bukan temuan valid). Prinsip anti-leakage ini non-negotiable.
