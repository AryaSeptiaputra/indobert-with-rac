# PROGRESS — IndoBERT-with-RAC

Status keseluruhan proyek (skripsi ITENAS 2026: trade-off performa vs efisiensi 3 strategi adaptasi IndoBERT untuk deteksi komentar judi). Terakhir diperbarui: **2026-07-25**.

## Ringkasan tahap

| Tahap | Status | Artefak |
|-------|--------|---------|
| EDA (`01_eda.ipynb`) | ✅ Selesai | `EDA_REPORT_BAGIAN1/2/3.md`, `results/figures/` |
| Preprocessing (`02_preprocessing.ipynb`) | ✅ Selesai | `dataset/splits/`, `dataset/processed/metadata.json`, `DATASET.md` |
| Sistem tuning UI (Vast.ai) | ✅ Selesai dibangun & dipakai | `app.py`, `src/job_runner.py`, `src/tuning.py`, `src/reporting.py`, `VAST_GUIDE.md` |
| **Tuning RM-a** | ✅ **Selesai** (26 run: grid 24 + coordinate descent 2) | `tuning_grids/RMA_TUNING_GRID.md` + 2 CSV tahap |
| **Tuning RM-b** | ✅ **Selesai** (27 run: arsitektur 4+2 + grid lr×epochs 15 + coordinate descent 6) | `tuning_grids/RMB_TUNING_GRID.md` + 4 CSV tahap |
| **Tuning RM-c** | ✅ **Selesai** (67 run: grid α×k 66 + cek weighting 1) | `tuning_grids/RMC_TUNING_GRID.md` + 2 CSV tahap |
| **Benchmark Final (TEST, satu sesi GPU)** | ✅ **Selesai** | `results/vast/metrics/{final_comparison,inference_benchmark,success_criteria}.csv` |

Seluruh rangkaian tuning + benchmark final **tuntas**. Sisa pekerjaan: penulisan Bab 4 skripsi
dan (opsional) tuning RM-c lanjutan bila ingin eksplorasi lebih jauh.

## Dataset final (input siap latih)
- `dataset/splits/{train,val,test}.csv` — kolom `textOriginal, text_clean, label`. **Input model = `text_clean`**.
- train 6.588 / val 1.402 / test 1.405 (total 9.395; ~18% kelas judi). Rasio ~4,5:1.
- Class weights (dari train): `{0: 0.611, 1: 2.752}`. Metrik utama **F1-macro**.
- Base model `indobenchmark/indobert-base-p2`, max_length 128, special token `[URL]/[MENTION]/[NUM]` → **model wajib `resize_token_embeddings`**.

## Hasil final 3 skenario (TEST set, satu sesi RTX 3090 — resmi untuk Bab 4)

| Model | Config final | F1-macro | F1 judi | Precision judi | Trainable params | Waktu latih | Latency inferensi |
|-------|------|----------|---------|----------------|------------------|-------------|---|
| RM-a (full fine-tune) | `lr=2e-5, epochs=5, batch=32, warmup=0,1, wd=0,01` | 0,9607 | 0,9357 | 0,9339 | 109.485.314 | 81,1 s | 9,40 ms |
| RM-b (frozen + MLP head) | `hidden_dim=1024, lr=1e-3, epochs=10, dropout=0,1, wd=0,0, batch=32` | 0,9486 | 0,9159 | 0,9176 | 789.506 (0,72%) | 11,65 s | 9,08 ms |
| RM-c (RAC) | `alpha=0,2, k=5, weighting=similarity` (head = RM-b di atas) | 0,9497 | 0,9176 | 0,9213 | 0 | 0,0 s | 10,96 ms |

**RM-b dan RM-c lolos 3/3 kriteria sukses** (gap F1 ≤3pp, reduksi param ≥90%, reduksi waktu ≥50% — syarat cuma 2/3): RM-b gap 1,21pp/reduksi param 99,28%/reduksi waktu 85,64%; RM-c gap 1,10pp/reduksi param 100%/reduksi waktu 100%.

**Temuan kunci:**
- RM-b: arsitektur **MLP jauh mengungguli linear** (dikonfirmasi ulang dari nol, bukan asumsi hasil lama), kapasitas optimal `hidden_dim=1024` (di atas dimensi input 768 — keputusan disengaja setelah tren kenaikan belum melandai sampai titik itu, dihentikan saat sinyal overfit pertama muncul).
- RM-c: RAC memberi perbaikan **kecil tapi konsisten arahnya** di atas RM-b murni (+0,11pp F1-macro test) — seluruhnya berasal dari perbaikan precision (mengurangi 1 false positive dari 1.405 sampel test), recall tak berubah sama sekali. Signifikansi statistik formal belum diuji (satu seed/split) — baca sebagai bonus tanpa risiko, bukan pendorong utama argumen kompetitif.
- **Latency inferensi RM-b/RM-c HAMPIR SAMA dengan RM-a** (bahkan RM-c sedikit lebih lambat) — efisiensi RM-b/RM-c ada di parameter & waktu **latih**, bukan kecepatan prediksi (forward pass encoder BERT 110M tetap penuh dijalankan di ketiganya saat inferensi).

## Eksplorasi encoder ringan (IndoBERT-lite) untuk RM-b/RM-c — infrastruktur siap, hasil awal

Motivasi: latency inferensi RM-b/RM-c hampir tidak membaik dibanding RM-a (lihat temuan kunci di
atas) karena encoder BERT-base 110M tetap dijalankan penuh saat inferensi. Untuk menyerang ini,
ditambahkan dukungan **ganti encoder** RM-b/RM-c ke varian yang lebih ringan:

- **Infrastruktur** (siap dipakai): `app.py` sidebar punya selectbox "Base encoder"
  (`indobert-base-p2` default vs `indobert-lite-base-p2`); `src/job_runner.py` menamai cache fitur
  per-model (`features/<model_slug>/`, `extract_meta.json` kini mencatat `model_name`+`hidden_dim`);
  `rmb_best.pt` mencatat `model_name` asalnya; RM-c menolak (raise error) jika `model_name` job
  beda dari encoder head RM-b yang dipakai (cegah kontaminasi silang). Kolom `model_name` kini ada
  di `runs_{rma,rmb,rmc}.csv`. Semua perubahan aditif & backward-compatible (checkpoint lama tanpa
  key ini tetap bisa dimuat, hanya validasi mismatch di-skip dengan aman).
- **Bug penting yang ditemukan & diperbaiki** (`src/dataset.py::load_tokenizer`): repo Hub
  `indobenchmark/indobert-lite-*` berarsitektur ALBERT tapi **hanya menyediakan vocab.txt
  WordPiece**, bukan file `.model` SentencePiece. `AutoTokenizer` salah menebak kelas
  (`AlbertTokenizer`, butuh SentencePiece) dan **diam-diam** jatuh ke vocab minimal 5-token (semua
  kata jadi `[UNK]`) — bukan error, sehingga bisa lolos tanpa disadari dan mencemari seluruh
  training/evaluasi. Diperbaiki dengan deteksi otomatis (`len(tokenizer) < 1000` → fallback ke
  `BertTokenizer`) + `sentencepiece` ditambahkan ke `requirements.txt` (tetap dibutuhkan agar
  `transformers` bisa mengenali kelas `AlbertTokenizer` sebelum fallback berjalan).
- **Validasi SMOKE (lokal, RTX 3050 Laptop, subset kecil, BUKAN angka final):** tokenizer (dengan
  fix di atas), `resize_token_embeddings`+seeding token khusus, ekstraksi fitur, training head RM-b,
  RAC RM-c, dan guard mismatch — **semua lolos** memakai `indobenchmark/indobert-lite-base-p2`.
- **Temuan awal (perbandingan latency satu-sesi/satu-GPU, encoder saja, sample tunggal):**
  encoder lite (ALBERT, 11,68 juta parameter) **TIDAK mempercepat forward pass** dibanding
  base-p2 (109,48 juta parameter) — 21,78 ms vs 21,41 ms (lite sedikit LEBIH LAMBAT). Yang turun
  drastis justru **peak GPU memory** (59,6 MB vs 433,4 MB, ~86% lebih kecil). Ini konsisten dengan
  arsitektur ALBERT (Lan et al., 2019): parameter berkurang lewat *cross-layer weight sharing*,
  bukan lewat pengurangan jumlah layer atau ukuran hidden — jumlah komputasi (FLOPs) per forward
  pass nyaris sama dengan BERT-base, jadi latency tidak ikut turun proporsional dengan parameter.
  **Implikasi:** `indobert-lite-base-p2` kemungkinan BUKAN jawaban untuk masalah latency RM-b/RM-c;
  ia menambah argumen efisiensi *memori*, bukan *kecepatan*. Untuk benar-benar mengejar latency,
  perlu varian dengan **lebih sedikit layer transformer** (mis. model ter-distilasi), bukan
  ALBERT-style parameter sharing.
- **Belum dilakukan** (di luar lingkup validasi ini, perlu keputusan berikutnya): kampanye tuning
  sungguhan dengan encoder lite (grid RM-b baru + turunan RM-c) di `out_dir` terpisah (mis.
  `results/vast_lite/`) di Vast.ai untuk angka F1/efisiensi yang valid Bab 4 — SMOKE run lokal di
  atas sudah dibersihkan (bukan artefak permanen).

## Sistem tuning UI — cara kerja

Dibangun untuk dijalankan di **Vast.ai** (GPU sewa) dengan alur **human-in-the-loop**, plus mode **Batch** (tambahan, aditif — single-config tetap ada) untuk menjalankan banyak konfigurasi sekaligus dari grid nilai (Cartesian) atau tabel CSV siap-pakai:

- **`app.py`** — Streamlit Control Panel (4 tab: Status / Tuning / Final / Hasil). Setiap run (single atau batch) menumpuk di `runs_{rma,rmb,rmc}.csv` + kolom `catatan`, `delta_vs_best_f1_macro_pp`, `is_tie_with_best`, `overfit_signal` (semua otomatis); `best.json` melacak juara-sejauh-ini; RM-c otomatis pakai head RM-b terbaik; tab Final = benchmark inferensi satu sesi + verdict kriteria.
- **`src/job_runner.py`** — worker (proses terpisah, tetap jalan walau tab ditutup). `run_batch()` menjalankan banyak config berurutan, isolasi kegagalan per-config (`runs_{scenario}_errors.csv`), stop graceful.
- **`src/tuning.py`** — engine training/eval per-config (`train_eval_rma/rmb`, `eval_rmc`).
- **`src/reporting.py`** — artefak visual otomatis: kurva training per-run, confusion matrix + PR curve (juara & final), heatmap grid, scatter trade-off performa-vs-efisiensi, bar chart top-config.
- Panduan operasional lengkap: **`VAST_GUIDE.md`**. Rancangan tuning per skenario:
  folder **`tuning_grids/`** (`RMA_TUNING_GRID.md`, `RMB_TUNING_GRID.md`, `RMC_TUNING_GRID.md`
  + CSV per tahap siap unggah).

## Aturan validitas untuk Bab 4
- Angka **efisiensi** (waktu latih, latency, peak memory) HANYA valid bila dari **satu hardware, satu sesi** — sudah dipenuhi lewat tab Final (`results/vast/metrics/inference_benchmark.csv`, RTX 3090, satu sesi 2026-07-25).
- Angka **performa (F1)** hardware-independent, boleh lintas-hardware bila perlu dibandingkan dengan hasil awal (notebook 03a/b/c, Colab).

## Catatan operasional
- Reproduksi lokal (opsional, tak dipakai untuk angka Bab 4): `run_local_training.py` (RTX 3050, batch 32 pakai grad-accum) → `results/local/` (saat ini kosong, belum dijalankan ulang pasca clean-slate).
- Bug proses zombie (`job_alive()` salah lapor job masih jalan padahal sudah selesai) sudah diperbaiki di `app.py` — lihat riwayat commit/percakapan untuk detail.
- Setelah seluruh hasil diunduh (`report_bundle.zip` dari tab Hasil), **DESTROY instance Vast.ai** (bukan Stop) sesuai `VAST_GUIDE.md` §6.
