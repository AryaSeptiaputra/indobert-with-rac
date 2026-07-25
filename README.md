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
ta-indobert-judi/
│
├── data/
│   ├── raw/                  # Data mentah dari Excel (jangan diubah)
│   ├── processed/            # Data setelah preprocessing
│   └── splits/               # Train / val / test set (70:15:15)
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_rma_finetune.ipynb
│   ├── 04_rmb_frozen.ipynb
│   └── 05_rmc_rac.ipynb
│
├── src/
│   ├── preprocessing.py      # Fungsi cleaning & normalisasi teks
│   ├── dataset.py            # PyTorch Dataset class
│   ├── model_rma.py          # Full fine-tuning pipeline
│   ├── model_rmb.py          # Frozen encoder pipeline
│   ├── model_rmc.py          # Frozen encoder + RAC pipeline
│   ├── rac.py                # Logit fusion & FAISS retrieval
│   └── evaluate.py           # Fungsi evaluasi & pengukuran efisiensi
│
├── results/
│   ├── metrics/              # CSV hasil evaluasi tiap skenario
│   ├── checkpoints/          # Model checkpoint terbaik
│   └── figures/              # Grafik & visualisasi hasil
│
├── DATASET.md                # Dokumentasi dataset
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

### Isi `requirements.txt`

```
torch==2.1.0
transformers==4.38.0
datasets==2.18.0
faiss-gpu==1.7.2        # ganti faiss-cpu jika tidak ada GPU
scikit-learn==1.4.0
pandas==2.2.0
openpyxl==3.1.2
numpy==1.26.4
matplotlib==3.8.0
seaborn==0.13.2
tqdm==4.66.0
```

> **Catatan Vast.ai:** Gunakan image PyTorch resmi (`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`) agar versi CUDA kompatibel.

---

## Cara Menjalankan

### Urutan eksekusi yang benar

Jalankan notebook sesuai urutan nomor:

```
01_eda → 02_preprocessing → 03_rma → 04_rmb → 05_rmc
```

Jangan lewati `02_preprocessing` — output-nya dipakai oleh semua notebook model.

### Menjalankan skenario individual (via script)

```bash
# RM-a: Full fine-tuning
python src/model_rma.py --epochs 5 --lr 2e-5 --batch_size 16

# RM-b: Frozen encoder
python src/model_rmb.py --epochs 5 --lr 2e-4 --batch_size 32

# RM-c: Frozen encoder + RAC
python src/model_rmc.py --epochs 5 --lr 2e-4 --batch_size 32 --alpha 0.3 --k 5
```

---

## Hyperparameter Default

| Parameter | RM-a | RM-b | RM-c |
|-----------|------|------|------|
| Base model | `indobenchmark/indobert-base-p2` | sama | sama |
| Epochs | 5 | 5 | 5 |
| Learning rate | 2e-5 | 2e-4 | 2e-4 |
| Batch size | 16 | 32 | 32 |
| Max token length | 128 | 128 | 128 |
| Alpha (α) RAC | — | — | 0.3 |
| k (nearest neighbors) | — | — | 5 |
| Optimizer | AdamW | AdamW | AdamW |
| Scheduler | Linear warmup | Linear warmup | Linear warmup |

> Hyperparameter α dan k pada RM-c dapat berubah setelah tuning. Catat hasilnya di `DATASET.md`.

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
| RM-a (full fine-tuning) | ✅ Selesai (baseline) |
| RM-b (frozen encoder) | ✅ Selesai (baseline) |
| RM-c (frozen + RAC) | ✅ Selesai (baseline) |
| Sistem tuning (UI Streamlit → Vast.ai) | 🟡 Siap, belum dijalankan |
| Tuning final + benchmark satu-sesi | ⬜ Belum (jalankan di Vast.ai) |
| Analisis hasil & penulisan | ⬜ Belum |

> Detail status & langkah berikutnya: lihat [`PROGRESS.md`](PROGRESS.md) dan [`VAST_GUIDE.md`](VAST_GUIDE.md).

---

## Referensi Utama

- Manullang et al. (2025) — JAIC, DOI: `10.30871/jaic.v9i3.9468`
- Devlin et al. (2019) — BERT, DOI: `10.18653/v1/N19-1423`
- Yu et al. (2023) — RAC for few-shot text classification, DOI: `10.18653/v1/2023.findings-emnlp.477`
- Tandi et al. (2025) — JISEBI, frozen IndoBERT untuk Textual Entailment
- Long et al. (2022) — CVPR, DOI: `10.1109/CVPR52688.2022.00683`