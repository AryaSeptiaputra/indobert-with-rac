# Analisis Trade-off Performa dan Efisiensi Komputasi pada Strategi Adaptasi IndoBERT dengan Retrieval-Augmented Classification untuk Deteksi Komentar Promosi Judi Daring

**Penulis:** Arya Eka Septiaputra (NRP 152022190)
**Program Studi:** Informatika — Institut Teknologi Nasional Bandung (ITENAS)
**Tahun:** 2026

---

## Deskripsi

Penelitian ini membandingkan tiga strategi adaptasi IndoBERT untuk mendeteksi
komentar promosi judi daring berbahasa Indonesia di YouTube, dengan fokus pada
trade-off antara performa klasifikasi dan efisiensi komputasi.

| Kode | Strategi | Deskripsi |
|------|----------|-----------|
| RM-a | Full Fine-tuning | Seluruh parameter IndoBERT diperbarui (baseline) |
| RM-b | Frozen Encoder | Hanya classification head yang dilatih |
| RM-c | Frozen Encoder + RAC | Encoder beku diperkuat retrieval berbasis FAISS |

Model dasar `indobenchmark/indobert-base-p2`, label biner (0 = normal,
1 = promosi judi), metrik utama F1-macro.

---

## Setup

Python 3.12 atau lebih baru.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements-dev.txt
pip install -e .
cp .env.example .env
```

`pip install -e .` mendaftarkan paket `src` sehingga `from src... import ...`
berfungsi dari notebook mana pun tanpa memanipulasi `sys.path`.

Untuk GPU, pasang torch dari index CUDA yang sesuai lebih dulu:

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
```

---

## Struktur

```
IndoBERT-with-RAC/
├── src/
│   ├── config.py              # Settings (pydantic-settings), seluruh path & default
│   ├── models/
│   │   ├── schemas.py         # RMAConfig / RMBConfig / RMCConfig / RunRequest
│   │   ├── comment_dataset.py # Tokenizer + PyTorch Dataset
│   │   └── heads.py           # Factory encoder dan classification head
│   ├── services/
│   │   ├── preprocessing.py   # TextCleaner, DatasetBuilder
│   │   ├── data.py            # ExperimentData (split + tokenizer + class weight)
│   │   ├── features.py        # FeatureExtractor, cache embedding beku
│   │   ├── training.py        # RMATrainer, RMBTrainer, RMCEvaluator
│   │   ├── rac.py             # RACClassifier (FAISS + fusi probabilitas)
│   │   ├── evaluation.py      # ClassificationEvaluator, EfficiencyProfiler
│   │   ├── campaign.py        # CampaignRunner (orkestrasi run dan benchmark)
│   │   ├── run_log.py         # RunLogger, HistoryWriter, BestTracker
│   │   ├── reporting.py       # FigureReporter
│   │   ├── faiss_benchmark.py # FaissBenchmark
│   │   ├── aggregation.py     # RunMerger
│   │   └── workbook.py        # WorkbookBuilder (ekspor Excel)
│   └── utils/                 # logger, seeding, I/O atomik
├── tests/                     # pytest, mirror struktur src/
├── notebooks/                 # ENTRY POINT seluruh pipeline
├── data/
│   ├── raw/                   # data_labeling.csv (jangan diubah)
│   ├── interim/               # data_clean.csv
│   └── processed/             # train/val/test.csv + metadata.json
├── models/                    # checkpoint final hasil ekspor
├── outputs/                   # keluaran eksperimen
├── tuning_grids/              # rancangan grid per skenario (input kampanye)
└── docs/                      # laporan EDA, referensi, arsip
```

---

## Struktur branch

Repo ini punya tiga branch dengan peran berbeda, supaya angka efisiensi dari
hardware yang berbeda tidak tercampur (lihat "Aturan validitas Bab 4" di
bawah):

| Branch | Isi |
|---|---|
| `main` | Codebase rujukan (kode saja). Tidak menyimpan hasil eksperimen apa pun — titik awal clone. |
| `local` | Hasil kampanye yang dijalankan di mesin lokal (RTX 3050 Laptop, 4 GB VRAM). |
| `vast.ai` | Hasil kampanye yang dijalankan di instance Vast.ai (RTX 3090). |

Untuk mulai kerja di instance Vast.ai:

```bash
git clone https://github.com/AryaSeptiaputra/indobert-with-rac.git
cd indobert-with-rac
git checkout vast.ai
```

Codebase (`src/`, `notebooks/`, `tests/`) identik di ketiga branch — yang
berbeda hanya isi `outputs/` dan narasi `PROGRESS.md`. Jangan gabungkan angka
efisiensi (waktu latih, latency, peak memory) dari `local` dan `vast.ai` dalam
satu tabel; F1-macro boleh dibandingkan lintas branch karena tidak bergantung
hardware.

---

## Cara menjalankan

Seluruh pipeline dijalankan dari notebook, berurutan:

```
01_eda → 02_preprocessing → 03a_rma → 03b_rmb → 03c_rmc
       → 04_tuning_campaign → 05_final_benchmark → 06_analysis_export
```

| Notebook | Isi |
|---|---|
| `01_eda.ipynb` | EDA; mengunci kunci dedup, `max_length`, class weight, placeholder |
| `02_preprocessing.ipynb` | Membangun split 70:15:15 beserta gate reproduktibilitas |
| `03a_rma_finetune.ipynb` | Baseline RM-a, satu konfigurasi |
| `03b_rmb_frozen.ipynb` | Baseline RM-b, ekstraksi fitur beku |
| `03c_rmc_rac.ipynb` | Baseline RM-c, sweep alpha |
| `04_tuning_campaign.ipynb` | Kampanye grid ketiga skenario |
| `05_final_benchmark.ipynb` | Split test dan benchmark inferensi, satu sesi |
| `06_analysis_export.ipynb` | Biaya FAISS, penggabungan riwayat, ekspor Excel |

Jangan lewati `02_preprocessing.ipynb`: seluruh notebook model bergantung pada
keluarannya.

Menjalankan test:

```bash
pytest
```

---

## Hyperparameter

Nilai awal (`src/models/schemas.py`) adalah titik masuk tuning, bukan nilai final.

| Parameter | RM-a | RM-b | RM-c |
|-----------|------|------|------|
| Epochs | 5 | 5 | — |
| Learning rate | 2e-5 | 2e-4 | — |
| Batch size | 16 | 32 | — |
| Warmup ratio | 0,1 | — | — |
| Weight decay | 0,01 | 0,01 | — |
| Head | — | linear (atau `mlp`, `hidden_dim` 256) | mewarisi head RM-b |
| Dropout | — | 0,1 | — |
| Alpha | — | — | 0,3 |
| k | — | — | 5 |
| Optimizer | AdamW | AdamW | — (tanpa training) |

Nilai final ditentukan lewat kampanye di `04_tuning_campaign.ipynb` dan dicatat
di `PROGRESS.md`.

---

## Mekanisme RAC (RM-c)

RAC menggabungkan dua **distribusi probabilitas** saat inferensi, bukan logit:

```
p_final = (1 - alpha) * softmax(head(embedding)) + alpha * p_retr
pred    = argmax(p_final)
```

`p_retr` adalah distribusi label dari k tetangga terdekat di indeks FAISS yang
dibangun HANYA dari split train. Softmax diterapkan sekali, di cabang BERT
sebelum fusi, dan tidak pernah setelahnya: kedua masukan sudah berupa distribusi
dan bobot fusinya berjumlah satu, sehingga hasilnya sudah sah. Softmax kedua akan
meratakan selisih dan bisa mengubah argmax pada kasus nyaris seri. Fusi level
probabilitas dan level logit tidak ekuivalen.

---

## Metrik evaluasi

**Klasifikasi:** accuracy, precision, recall, F1 (macro dan weighted), confusion
matrix. Data timpang sekitar 4,5:1, sehingga metrik utama F1-macro dengan F1
kelas judi sebagai pemecah seri.

**Efisiensi:** jumlah trainable parameter, waktu latih, peak GPU memory, dan
latency inferensi per sampel.

**Kriteria sukses:** strategi ringan dianggap kompetitif bila memenuhi minimal
dua dari tiga syarat berikut.

- Selisih F1 tidak lebih dari 3 poin persentase terhadap RM-a
- Pengurangan trainable parameter minimal 90%
- Pengurangan waktu latih minimal 50%

---

## Aturan validitas Bab 4

Angka efisiensi (waktu latih, latency, peak memory) hanya sah bila diukur pada
**satu hardware dan satu sesi**. Jangan mencampur angka dari mesin atau sesi
berbeda dalam satu tabel efisiensi. F1 tidak bergantung hardware dan boleh
dibandingkan lintas mesin.

---

## Catatan penyimpangan dari standar `writer-code`

- **Tidak ada folder `scripts/`.** Seluruh pipeline dipanggil dari notebook,
  sehingga notebook berperan sebagai entry point.
- **Tidak ada layer API.** Proyek ini penelitian, bukan aplikasi berbackend.

---

## Referensi utama

- Manullang dkk. (2025) — JAIC, DOI: `10.30871/jaic.v9i3.9468`
- Devlin dkk. (2019) — BERT, DOI: `10.18653/v1/N19-1423`
- Wilie dkk. (2020) — IndoNLU
- Yu dkk. (2023) — RAC untuk few-shot text classification, DOI: `10.18653/v1/2023.findings-emnlp.477`
- Tandi dkk. (2025) — JISEBI, frozen IndoBERT untuk Textual Entailment
- Lan dkk. (2019) — ALBERT

Daftar lengkap: `docs/DAFTAR_REFERENSI.pdf`.
