# Analisis Trade-off Performa dan Efisiensi Komputasi pada Strategi Adaptasi IndoBERT dengan Retrieval-Augmented Classification untuk Deteksi Komentar Promosi Judi Daring

**Penulis:** Arya Eka Septiaputra (NRP 152022190)  
**Program Studi:** Informatika — Institut Teknologi Nasional Bandung (ITENAS)  
**Tahun:** 2026

---

## Deskripsi Proyek

Penelitian ini mengadaptasi model IndoBERT dengan strategi adaptasi ringan, yaitu Frozen Encoder dan Frozen Encoder + Retrieval-Augmented Classification (RAC), untuk domain deteksi komentar promosi judi daring berbahasa Indonesia di YouTube. Penelitian ini membandingkan tiga strategi adaptasi internal IndoBERT — Full Fine-Tuning, Frozen Encoder, dan Frozen Encoder + RAC — dengan tujuan menganalisis trade-off antara performa klasifikasi dan efisiensi komputasi.

Tiga skenario yang dibandingkan:

| Kode | Strategi | Deskripsi |
|------|----------|-----------|
| RM-a | Full Fine-tuning | Seluruh parameter IndoBERT diperbarui (baseline) |
| RM-b | Frozen Encoder | Hanya classification head yang dilatih |
| RM-c | Frozen Encoder + RAC | Frozen encoder diperkuat retrieval berbasis FAISS |

---

## Struktur Folder

```
IndoBERT-with-RAC/
│
├── dataset/
│   ├── raw/                  # Data mentah (data_labeling.csv, jangan diubah)
│   ├── processed/            # Data setelah preprocessing + metadata.json
│   └── splits/               # Train / val / test set (70:15:15)
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03a_rma_finetune.ipynb          # Baseline RM-a (full fine-tuning)
│   ├── 03b_rmb_frozen.ipynb            # Baseline RM-b (frozen encoder)
│   ├── 03c_rmc_rac.ipynb               # Baseline RM-c (frozen + RAC)
│   └── 04a_*_tuning_*.ipynb / 04b_*_tuning_*.ipynb   # Legacy, superseded by app.py
│
├── src/
│   ├── preprocessing.py      # Fungsi cleaning & normalisasi teks
│   ├── dataset.py            # PyTorch Dataset class
│   ├── modeling.py           # Model/head factories (RM-a/b/c)
│   ├── tuning.py             # Engine training/eval per-config (1 call = 1 config)
│   ├── job_runner.py         # Subprocess worker untuk app.py (fase rma/rmb/rmc/final)
│   ├── rac.py                # Fusi probabilitas & FAISS retrieval
│   ├── reporting.py          # Figur otomatis (kurva, heatmap grid, tradeoff scatter)
│   └── evaluate.py           # Fungsi evaluasi & pengukuran efisiensi
│
├── results/
│   ├── vast/                 # Output tuning Vast.ai (runs_*.csv, best.json, checkpoints/, metrics/, figures/)
│   ├── local/                # Output run_local_training.py
│   └── figures/              # Figur EDA
│
├── app.py                    # Streamlit Tuning Control Panel (entry point utama)
├── run_local_training.py     # Reproduksi lokal RM-a/b/c end-to-end
├── DATASET.md                # Dokumentasi dataset
├── PROGRESS.md               # Status terkini proyek (sumber kebenaran)
├── README.md                 # File ini
└── requirements.txt          # Daftar dependensi Python
```

---

## Setup Environment

### 1. Clone / salin proyek

```bash
git clone <repo-url>
cd ta-indobert-judi
```

### 2. Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# atau
venv\Scripts\activate           # Windows
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

### Isi `requirements.txt` (ringkasan)

```
torch==2.1.0
transformers==4.38.0
datasets==2.18.0
faiss-cpu==1.7.4        # pakai faiss-gpu bila tersedia
scikit-learn==1.4.0
pandas==2.2.0
openpyxl==3.1.2
numpy==1.26.4
matplotlib==3.8.0
seaborn==0.13.2
tqdm==4.66.0
langdetect==1.0.9
streamlit>=1.30          # dibutuhkan app.py (tuning control panel)
sentencepiece            # dibutuhkan transformers utk mengenali AlbertTokenizer; indobert-lite-*
                          # sebenarnya fallback ke BertTokenizer (vocab.txt) -- lihat src/dataset.py
```

> **Catatan Vast.ai:** image harus membawa **torch ≥ 2.4** (dibutuhkan `torch.amp` di `src/tuning.py`) dengan CUDA 12.x — lihat `VAST_GUIDE.md`.

---

## Cara Menjalankan

### Urutan eksekusi yang benar

Jalankan notebook sesuai urutan nomor:

```
01_eda → 02_preprocessing → 03a_rma_finetune → 03b_rmb_frozen → 03c_rmc_rac
```

Jangan lewati `02_preprocessing` — output-nya dipakai oleh semua notebook model. Notebook `04a_*`/`04b_*` bersifat legacy, sudah digantikan oleh `app.py`.

### Menjalankan tuning (via app.py)

Tidak ada script CLI per-skenario — seluruh training/tuning berjalan lewat Streamlit control panel (`app.py`, via `src/job_runner.py`) atau lewat `run_local_training.py` untuk reproduksi lokal end-to-end:

```bash
# Tuning control panel (workflow utama — jalankan di instance Vast.ai)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Reproduksi lokal semua skenario RM-a/b/c
python run_local_training.py
```

---

## Hyperparameter

### Nilai awal tuning (starting defaults, `src/tuning.py`)

| Parameter | RM-a | RM-b | RM-c |
|-----------|------|------|------|
| Base model | `indobenchmark/indobert-base-p2` | sama | sama |
| Epochs | 5 | 5 | — |
| Learning rate | 2e-5 | 2e-4 | — |
| Batch size | 16 | 32 | — |
| Max token length | 128 | 128 | 128 |
| Alpha (α) RAC | — | — | 0.3 |
| k (nearest neighbors) | — | — | 5 |
| Optimizer | AdamW | AdamW | — |
| Scheduler | Linear warmup | Linear warmup | — |

### Konfigurasi final (hasil tuning Vast.ai, 2026-07-25 — lihat `PROGRESS.md`)

| Parameter | RM-a | RM-b | RM-c |
|-----------|------|------|------|
| Learning rate | 2e-5 | 1e-3 | — |
| Epochs | 5 | 10 | — |
| Batch size | 32 | 32 | — |
| Head architecture | — | `mlp`, hidden_dim=1024 | inherits RM-b head |
| Dropout | — | 0.1 | — |
| Weight decay | 0.01 | 0.0 | — |
| Alpha (α) RAC | — | — | **0.2** |
| k (nearest neighbors) | — | — | **5** |
| weighting | — | — | similarity |

---

## Metrik Evaluasi

### Performa Klasifikasi
- Accuracy, Precision, Recall, F1-score (macro & weighted)
- Confusion matrix

### Efisiensi Komputasi
- Jumlah trainable parameters
- Training time (detik per epoch & total)
- GPU memory usage (MB) — peak saat training
- Inference latency (ms per sampel)

Semua hasil tersimpan otomatis di `results/metrics/`.

---

## Kriteria Sukses

Strategi ringan (RM-b atau RM-c) dianggap **kompetitif** terhadap full fine-tuning (RM-a) jika memenuhi minimal **2 dari 3** syarat berikut:

- Selisih F1-score ≤ 3 poin persentase
- Pengurangan trainable parameters ≥ 90%
- Pengurangan training time ≥ 50%

---

## Status Eksperimen

| Tahap | Status |
|-------|--------|
| Pengumpulan & pelabelan data | ✅ Selesai |
| EDA | ✅ Selesai |
| Preprocessing | ✅ Selesai |
| RM-a (full fine-tuning) — baseline notebook | ✅ Selesai |
| RM-b (frozen encoder) — baseline notebook | ✅ Selesai |
| RM-c (frozen + RAC) — baseline notebook | ✅ Selesai |
| Sistem tuning (UI Streamlit → Vast.ai) | ✅ Selesai dibangun & dipakai |
| Tuning RM-a/RM-b/RM-c (Vast.ai) | ✅ Selesai (26+31+67 run, 2026-07-25) |
| Tuning final + benchmark satu-sesi (TEST) | ✅ Selesai (RTX 3090, 2026-07-25) |
| Analisis hasil & penulisan Bab 4 | 🟡 Berjalan |

> Detail status & langkah berikutnya: lihat [`PROGRESS.md`](PROGRESS.md) dan [`VAST_GUIDE.md`](VAST_GUIDE.md).

---

## Referensi Utama

- Manullang et al. (2025) — JAIC, DOI: `10.30871/jaic.v9i3.9468`
- Devlin et al. (2019) — BERT, DOI: `10.18653/v1/N19-1423`
- Yu et al. (2023) — RAC for few-shot text classification, DOI: `10.18653/v1/2023.findings-emnlp.477`
- Tandi et al. (2025) — JISEBI, frozen IndoBERT untuk Textual Entailment
- Long et al. (2022) — CVPR, DOI: `10.1109/CVPR52688.2022.00683`