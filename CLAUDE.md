# CLAUDE.md

Panduan untuk Claude Code saat bekerja di repositori ini.

## Ikuti standar `writer-code`

Seluruh kode Python di repo ini mengikuti skill `writer-code`. Invoke skill itu
sebelum menulis, mereview, atau merefaktor kode apa pun di sini.

## Gambaran proyek

Skripsi S1 (ITENAS 2026) yang membandingkan tiga strategi adaptasi IndoBERT untuk
mendeteksi komentar promosi judi daring berbahasa Indonesia di YouTube.
Pertanyaan pusatnya adalah trade-off antara performa dan efisiensi:

| Kode | Strategi | Deskripsi |
|------|----------|-----------|
| RM-a | Full Fine-tuning | Seluruh parameter IndoBERT diperbarui (baseline) |
| RM-b | Frozen Encoder | Hanya classification head yang dilatih |
| RM-c | Frozen Encoder + RAC | Encoder beku diperkuat retrieval FAISS |

Model dasar `indobenchmark/indobert-base-p2`. Label biner (0 = normal,
1 = promosi judi). Metrik utama F1-macro.

## Lingkungan

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
```

`pip install -e .` mendaftarkan paket `src`. Repo ini TIDAK memakai
`sys.path.insert` di mana pun; kalau menemukannya, itu regresi.

Python 3.13, torch 2.12.1, transformers 5.12.1 (lihat `requirements.txt` untuk
pin lengkap — versinya adalah yang benar-benar terpasang dan tervalidasi).

## Menjalankan

Seluruh pipeline dipanggil dari notebook. Tidak ada `scripts/`, tidak ada CLI,
tidak ada UI.

```
01_eda → 02_preprocessing → 03a_rma → 03b_rmb → 03c_rmc
       → 05_final_benchmark → 06_analysis_export → 04_bab4_artifacts
```

Jangan lewati `02_preprocessing.ipynb`: notebook model bergantung padanya.

03a sampai 03c adalah kampanye sesungguhnya, satu notebook per skenario, dalam
satu mesin dan berurutan: 03a grid RM-a, 03b grid RM-b, 03c grid RM-c (seluruh
head RM-b x alpha x k). Ketiganya menulis ke `outputs/tuning/`, folder yang juga
dibaca 05 dan 06. Tahap RM-b menyimpan state setiap head di
`checkpoints/rmb_heads/`; RM-c memuat head itu, tidak melatih ulang.

`04_bab4_artifacts.ipynb` membangkitkan Gambar 4.1-4.8 dari log kampanye
(`src/services/thesis_figures.py`, keluaran `figures/bab4/`). Gambar 4.1-4.4
cukup butuh 03a-03c; gambar 4.5-4.8 butuh 05, jadi jalankan 04 lagi setelah 05.

```bash
pytest        # 376 test, tanpa GPU
```

## Arsitektur

### Alur data

```
data/raw/data_labeling.csv
    → src/services/preprocessing.py   (TextCleaner, DatasetBuilder)
    → data/interim/data_clean.csv
    → data/processed/{train,val,test}.csv + metadata.json
    → src/services/data.py            (ExperimentData)
    → src/services/features.py        (FeatureExtractor, RM-b/RM-c)
    → src/services/training.py        (RMATrainer, RMBTrainer, RMCEvaluator)
    → outputs/tuning/
```

### Modul

| Modul | Isi |
|---|---|
| `src/config.py` | `Settings` berbasis pydantic-settings. SATU-SATUNYA tempat path, nama model, seed, dan default hidup |
| `src/models/schemas.py` | `RMAConfig`, `RMBConfig`, `RMCConfig`, `RunRequest` |
| `src/models/comment_dataset.py` | `load_tokenizer` (dengan fallback WordPiece), `GamblingCommentDataset` |
| `src/models/heads.py` | `build_finetune_model`, `build_encoder`, `build_head`, `mean_pool` |
| `src/services/preprocessing.py` | `TextCleaner`, `DatasetBuilder` |
| `src/services/data.py` | `ExperimentData` |
| `src/services/features.py` | `FeatureExtractor`, `FeatureSet` |
| `src/services/training.py` | `RMATrainer`, `RMBTrainer`, `RMCEvaluator` |
| `src/services/rac.py` | `RACClassifier`, `NeighborCache` |
| `src/services/fusion_ablation.py` | `FusionFormulaComparator`: evaluator fusi RM-c (kampanye hanya memakai `linear`; Rumus 1-4 masih ada di kode tetapi tidak dipakai) |
| `src/services/evaluation.py` | `ClassificationEvaluator`, `EfficiencyProfiler` |
| `src/services/campaign.py` | `CampaignRunner` — orkestrasi run, batch, benchmark final |
| `src/services/run_log.py` | `RunLogger`, `HistoryWriter`, `BestTracker` |
| `src/services/reporting.py` | `FigureReporter` |
| `src/services/thesis_figures.py` | `ThesisFigureBuilder`: Gambar 4.1-4.8 Bab 4 (PNG 300 dpi + PDF) langsung dari log |
| `src/services/faiss_benchmark.py` | `FaissBenchmark` |
| `src/services/aggregation.py` | `RunMerger` |
| `src/services/workbook.py` | `WorkbookBuilder` |
| `src/services/archive.py` | `ResultArchiver`: bungkus `outputs/` dan `HASIL.xlsx` menjadi `hasil_*.tar.gz` untuk diunduh |
| `src/utils/` | `setup_logger`, `set_seed`, I/O atomik |

### Mekanisme RAC (RM-c)

RAC menggabungkan dua **distribusi probabilitas** saat inferensi, bukan logit:

```
p_bert  = softmax(head(embedding))
p_retr  = distribusi label k tetangga terdekat (indeks FAISS HANYA dari train)
p_final = (1 - alpha) * p_bert + alpha * p_retr
pred    = argmax(p_final)
```

**Softmax diterapkan sekali, di cabang BERT sebelum fusi, tidak pernah
sesudahnya.** Kedua masukan sudah berupa distribusi dan bobot fusi berjumlah
satu, jadi `p_final` sudah sah; softmax kedua akan meratakan selisih dan bisa
mengubah argmax pada kasus nyaris seri. Saat menulis Bab 4, nyatakan fusinya di
level probabilitas — fusi level probabilitas dan level logit tidak ekuivalen.

Itu SATU-SATUNYA rumus RM-c. Rumus 1-4
(fusi level skor, alpha adaptif, geometric pooling) sempat diuji lalu dikeluarkan
dari kampanye (2026-09-24); kodenya masih di `fusion_ablation.py` tetapi tidak
masuk grid mana pun.

## Aturan yang tidak boleh dilanggar

1. **Angka efisiensi hanya sah dari satu hardware dan satu sesi.** Waktu latih,
   latency, dan peak memory tidak boleh dicampur lintas mesin atau sesi dalam
   satu tabel. F1 tidak bergantung hardware.
2. **Split test hanya dibuka di `05_final_benchmark.ipynb`.** Seleksi
   hyperparameter memakai validation. `eval_test` default False; jangan
   menyalakannya selama tuning.
3. **Indeks FAISS hanya dari split train.** Memasukkan val atau test membuat
   retrieval menemukan sampel uji di dalam indeksnya sendiri.
4. **Jangan menimpa `data/processed/` tanpa gate checksum.** Notebook 02 punya
   sel yang membandingkan checksum ISI split baru dengan yang lama
   (`DatasetBuilder.verify_reproducibility`) dan menolak melanjutkan bila
   berbeda. Yang dibandingkan isi kanonik, bukan SHA-256 byte berkas: hash byte
   berbeda antar OS (CRLF vs LF) dan antar git `autocrlf`. Split yang berubah
   membuat seluruh riwayat run menjadi yatim.
5. **Input model adalah kolom `text_clean`**, bukan `textOriginal`.
6. **Model wajib `resize_token_embeddings`** karena preprocessing menambahkan
   `[URL]`, `[MENTION]`, `[NUM]`. Sudah ditangani factory di `heads.py`.

## Konvensi hasil

`CampaignRunner` menulis ke `out_dir` (default `outputs/tuning/`). Seluruh
`outputs/` di-gitignore (kecuali figur EDA dan README): hasil run tidak disimpan di
git, karena `runs_*.csv` dan `best.json` yang ikut ter-clone membuat kampanye baru
melewati semua konfigurasi (resume) dan tidak menghasilkan checkpoint juara. Amankan
hasil dengan sel "Arsipkan hasil" di `06_analysis_export.ipynb` (`ResultArchiver`)
sebelum instance dihancurkan.

- `runs_{rma,rmb,rmc}.csv` — satu baris per run, menumpuk lintas sesi. Memuat
  kolom `catatan` (alasan konfigurasi dicoba) dan kolom turunan
  `delta_vs_best_f1_macro_pp`, `is_tie_with_best`, `overfit_signal`.
- `runs_{scenario}_errors.csv` — konfigurasi yang gagal, diisolasi agar batch
  tidak batal.
- `best.json` — juara per skenario menurut F1-macro validation.
- `history/{scenario}_history.csv` — kurva per-epoch, satu berkas menumpuk per
  skenario dengan kolom `run_id`.
- `checkpoints/`, `figures/`, `metrics/`, `tuning_summary.json`, `hardware.json`.
  Checkpoint TIDAK ikut git (di-gitignore) sedangkan `best.json` ikut; di clone atau
  instance baru pulihkan dengan `runner.restore_checkpoints()`, karena menjalankan
  ulang skenario tidak membuat checkpoint juara kembali (F1 sama tidak dipromosikan).
  `checkpoints/rmb_heads/run_{id}.pt` menyimpan SETIAP head RM-b (bukan hanya
  juara). `checkpoints/rmc_best.pt` memuat head yang dipakai juara RM-c beserta
  konfigurasi fusinya.

`outputs/_archive_*/` berisi hasil kampanye Vast.ai lama. Disimpan sebagai jalan
mundur sampai kampanye lokal terbukti berhasil, lalu dihapus. Jangan dipakai
sebagai sumber angka Bab 4.

## Rancangan eksplorasi hyperparameter

**Metode hibrida** — grid kombinatorial untuk sumbu yang saling terkait,
coordinate descent untuk yang independen. Pembagian ini berasal dari mekanika
sesungguhnya di `src/services/training.py`: `steps = ceil(N/batch) x epochs`,
warmup adalah RASIO dari total langkah sehingga menormalkan dirinya sendiri, dan
weight decay AdamW memang terpisah dari gradien.

- **Grup A, digrid bersama:** `lr x epochs x batch`. Ketiganya bersama-sama
  menentukan lintasan optimasi — batch 32 pada 5 epoch memberi separuh jumlah
  langkah pembaruan dibanding batch 16, sehingga mengubah satu menggeser optimum
  yang lain.
- **Grup B, coordinate descent di sel pemenang:** `warmup_ratio` dan
  `weight_decay`.

**Aturan seleksi:** `val_f1_macro` utama, `val_f1_judi` sebagai pemecah seri
(kelas minoritas yang bisa tertutupi rata-rata makro). Selisih di bawah 0,15 pp
dihitung seri (satu seed) sehingga konfigurasi yang lebih murah menang. Baca grid
sebagai permukaan: pivot `lr x epochs` per `batch` memperlihatkan apakah lr
optimal ikut bergeser. Pemenang di tepi grid adalah sinyal untuk melebarkan
rentang, bukan untuk mengunci.

Rancangan lengkap ada di `tuning_grids/` (`.md` untuk penalaran, `.csv` siap
dimuat notebook 03a-03c).

**RM-c adalah satu grid: seluruh head RM-b x `alpha x k`.** Head dipilih lewat
`rmb_run_id` di `RMCConfig` (kosong berarti juara RM-b), dan 66 konfigurasi
`alpha x k` di `RMC_TUNING_GRID.csv` diterapkan ke setiap head; setiap kombinasi
satu baris `runs_rmc.csv`. Tahap 2 mencoba `weighting=uniform` sekali di sel juara.
Dengan ratusan kandidat, pemenang mentah rawan bias seleksi: baca `is_tie_with_best`
dan kenaikan per head terhadap `alpha=0`, bukan hanya juara mekanis. RM-c tidak
melatih apa pun, tetapi biayanya adalah biaya head yang dipakainya (bukan nol)
ditambah retrieval saat inferensi.

## Dataset

`data/raw/data_labeling.csv` — komentar YouTube berlabel. Kolom yang dipakai
`textOriginal` dan `label`; sisanya metadata dan sebagiannya memuat identitas.

`data/processed/` berisi split 70:15:15 (train 6.588 / val 1.402 / test 1.405,
total 9.395, sekitar 18% kelas judi, rasio 4,5:1). Class weight ada di
`metadata.json` (`{0: 0.611, 1: 2.752}`) dan dipakai pada `CrossEntropyLoss`.
Karena timpang, laporkan F1-macro dan F1 kelas judi, jangan accuracy saja.

## Kriteria sukses

Strategi ringan (RM-b atau RM-c) dianggap kompetitif bila memenuhi minimal dua
dari tiga syarat:

- Selisih F1 tidak lebih dari 3 poin persentase terhadap RM-a
- Pengurangan trainable parameter minimal 90%
- Pengurangan waktu latih minimal 50%

## Struktur branch

Repo ini punya tiga branch: `main` (codebase rujukan, tanpa data hasil run),
`local` (kampanye di mesin lokal, RTX 3050 Laptop), dan `vast.ai` (kampanye di
instance Vast.ai, RTX 3090). Saat bekerja di `local` atau
`vast.ai`, Aturan #1 di atas berlaku per branch: jangan campur angka efisiensi
dari kedua branch itu dalam satu tabel Bab 4. Codebase identik di ketiga
branch; yang berbeda hanya narasi `PROGRESS.md`. Hasil run tidak disimpan di git.

## Status

Lihat `PROGRESS.md` sebagai sumber kebenaran.
