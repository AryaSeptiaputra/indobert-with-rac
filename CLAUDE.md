# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project (undergraduate thesis, ITENAS 2026) comparing three IndoBERT adaptation strategies for detecting Indonesian online gambling promotion comments on YouTube. The central question is the performance-vs-efficiency trade-off between:

| Code | Strategy | Description |
|------|----------|-------------|
| RM-a | Full Fine-tuning | All IndoBERT parameters updated (baseline) |
| RM-b | Frozen Encoder | Only the classification head is trained |
| RM-c | Frozen Encoder + RAC | Frozen encoder augmented with FAISS-based retrieval |

Base model: `indobenchmark/indobert-base-p2`. Labels are binary (0 = non-gambling, 1 = gambling promotion).

## Environment Setup

The `.venv` directory is the active virtual environment. Activate it before running anything:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies (not yet present — refer to README for `requirements.txt` contents):

```bash
pip install torch transformers datasets faiss-gpu scikit-learn pandas openpyxl numpy matplotlib seaborn tqdm
```

Use `faiss-cpu` if no GPU is available. On Vast.ai, use the `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` image.

## Running the Project

### Notebooks (primary workflow)

Run in strict order — each notebook's output feeds the next:

```
01_eda.ipynb → 02_preprocessing.ipynb → 03_rma_finetune.ipynb → 04_rmb_frozen.ipynb → 05_rmc_rac.ipynb
```

Do not skip `02_preprocessing.ipynb`; all model notebooks depend on its output in `dataset/processed/`.

### Individual model scripts

```bash
# RM-a: Full fine-tuning
python src/model_rma.py --epochs 5 --lr 2e-5 --batch_size 16

# RM-b: Frozen encoder
python src/model_rmb.py --epochs 5 --lr 2e-4 --batch_size 32

# RM-c: Frozen encoder + RAC
python src/model_rmc.py --epochs 5 --lr 2e-4 --batch_size 32 --alpha 0.3 --k 5
```

## Architecture

### Data flow

```
dataset/raw/data_labeling.csv
    → src/preprocessing.py   (cleaning, normalization)
    → dataset/processed/     (cleaned CSV)
    → dataset/splits/        (train/val/test, 70:15:15 split)
    → src/dataset.py         (PyTorch Dataset wrapper)
    → model pipelines
    → results/
```

### Source modules (to be created in `src/`)

- `preprocessing.py` — text cleaning and normalization for Indonesian text
- `dataset.py` — PyTorch `Dataset` class; tokenizes with IndoBERT tokenizer, max 128 tokens
- `model_rma.py` — full fine-tuning pipeline
- `model_rmb.py` — frozen encoder pipeline (only classification head trained)
- `model_rmc.py` — frozen encoder + RAC pipeline
- `rac.py` — FAISS index construction, k-NN retrieval, and logit fusion (`alpha` controls blend between BERT logits and retrieval-based logits)
- `evaluate.py` — classification metrics + computational efficiency measurements

### RAC mechanism (RM-c)

RAC fuses two probability distributions at inference time:
- BERT logits from the frozen encoder's classification head
- Retrieval-based distribution from k nearest neighbors in FAISS index (built from training embeddings)

The fusion weight `alpha` (default 0.3) blends them: `final_logits = (1 - alpha) * bert_logits + alpha * retrieval_logits`. The `k` parameter (default 5) controls neighbors used.

## Hyperparameters

| Parameter | RM-a | RM-b | RM-c |
|-----------|------|------|------|
| Epochs | 5 | 5 | 5 |
| Learning rate | 2e-5 | 2e-4 | 2e-4 |
| Batch size | 16 | 32 | 32 |
| Max token length | 128 | 128 | 128 |
| Alpha (α) | — | — | 0.3 |
| k (neighbors) | — | — | 5 |
| Optimizer | AdamW | AdamW | AdamW |
| Scheduler | Linear warmup | Linear warmup | Linear warmup |

Alpha and k in RM-c are tunable — document final values in `DATASET.md` after tuning.

## Results Convention

All outputs are saved automatically under `results/`:
- `results/metrics/` — CSV files with per-epoch and final evaluation metrics
- `results/checkpoints/` — best model checkpoint per strategy
- `results/figures/` — plots and confusion matrices

## Evaluation Metrics

**Classification:** Accuracy, Precision, Recall, F1-score (macro & weighted), confusion matrix.

**Efficiency:** Trainable parameter count, training time (seconds/epoch and total), peak GPU memory (MB), inference latency (ms/sample).

**Success criteria** — a lightweight strategy (RM-b or RM-c) is considered competitive if it meets at least 2 of 3:
- F1-score gap ≤ 3 percentage points vs. RM-a
- Trainable parameter reduction ≥ 90%
- Training time reduction ≥ 50%

## Dataset

`dataset/raw/data_labeling.csv` — labeled YouTube comments (Indonesian). Key columns: `textOriginal` (raw comment text), `label` (0 = normal, 1 = gambling promotion). All other columns (commentId, author, timestamps, etc.) are metadata and can be dropped during preprocessing.

`dataset/processed/` and `dataset/splits/` are empty and will be populated by `02_preprocessing.ipynb`.

## Current Status

Data collection and labeling are complete. EDA, preprocessing, and all modeling stages have not been started yet. Begin with `01_eda.ipynb`.
