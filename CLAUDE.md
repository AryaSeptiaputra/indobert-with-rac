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

Install dependencies (`requirements.txt` exists at the repo root):

```bash
pip install -r requirements.txt
```

Use `faiss-cpu` if no GPU is available. On Vast.ai, the image **must ship torch ≥ 2.4** (required by `torch.amp`, used in `src/tuning.py`) with a CUDA 12.x runtime — see `VAST_GUIDE.md`. The tuning UI additionally needs `streamlit`.

## Running the Project

### Notebooks (data prep + baselines)

Run in strict order — each notebook's output feeds the next:

```
01_eda.ipynb → 02_preprocessing.ipynb → 03a_rma_finetune.ipynb → 03b_rmb_frozen.ipynb → 03c_rmc_rac.ipynb
```

Do not skip `02_preprocessing.ipynb`; all model notebooks depend on its output in `dataset/processed/`.

`notebooks/04a_*_tuning_*.ipynb` and `04b_*_tuning_*.ipynb` are legacy — **superseded by the Streamlit app** (`app.py`) described below. Do not extend them.

### Entry points

```bash
# Tuning control panel (PRIMARY workflow — run this on the Vast.ai instance)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Standalone local reproduction of all three strategies (RM-a/b/c end to end)
python run_local_training.py
```

There are no per-model CLI scripts; all training goes through `app.py` (via `src/job_runner.py`) or `run_local_training.py`.

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

### Source modules (all present in `src/`)

- `preprocessing.py` — text cleaning and normalization for Indonesian text
- `dataset.py` — PyTorch `Dataset` class; tokenizes with IndoBERT tokenizer, max 128 tokens
- `modeling.py` — model/head factories: `build_finetune_model` (RM-a), `build_encoder` (frozen, RM-b/c), `mean_pool`, `extract_features`, `FrozenHead` (linear), `MLPHead`, `build_head`
- `tuning.py` — single-config training/eval engine: `train_eval_rma`, `train_eval_rmb`, `eval_rmc`, `RunLogger`, `test_rma`, and the starting defaults `RMA_DEFAULT` / `RMB_DEFAULT` / `RMC_DEFAULT`. **Contains no automatic search** — one call = one config.
- `job_runner.py` — subprocess worker; dispatches phases `rma` / `rmb` / `rmc` / `final`, writes progress, CSVs, checkpoints, and the final benchmark
- `rac.py` — FAISS index construction, k-NN retrieval, and logit fusion (`alpha` blends head probabilities with retrieval-based ones)
- `evaluate.py` — classification metrics + computational efficiency measurements

Repo-root scripts: `app.py` (Streamlit control panel) and `run_local_training.py` (standalone local run).

### RAC mechanism (RM-c)

RAC fuses two probability distributions at inference time:
- BERT logits from the frozen encoder's classification head
- Retrieval-based distribution from k nearest neighbors in FAISS index (built from training embeddings)

The fusion weight `alpha` (default 0.3) blends them: `final_logits = (1 - alpha) * bert_logits + alpha * retrieval_logits`. The `k` parameter (default 5) controls neighbors used.

## Tuning workflow (Streamlit → Vast.ai)

`app.py` is a Streamlit control panel with 4 tabs (Status / Tuning / Final / Hasil). The **Tuning** tab holds a scenario dropdown (RM-a / RM-b / RM-c) that swaps in the matching hyperparameter form; entered values persist when switching scenarios. It is built around **"one click = one configuration"** — there is deliberately **no automatic search**. The exploration strategy is decided by the human, with Claude analyzing each result to propose the next config.

- Each run is launched as a **separate subprocess** (`src/job_runner.py`), so it survives closing the browser tab; the UI only reads `progress.json` and the CSVs.
- The Tuning tab also has an additive **batch mode** (expander below the single-config Run button): a Cartesian value-list grid builder, or CSV paste/upload (e.g. `tuning_grids/RMA_TUNING_GRID.csv`, a machine-readable mirror of `tuning_grids/RMA_TUNING_GRID.md`'s Tahap 1 grid) — submits many configs as one job, still one subprocess, one row per config in the same CSVs. Single-config remains the default; batch never replaces it. Failed configs are isolated to `runs_{scenario}_errors.csv` and don't abort the batch. `src/reporting.py` auto-generates grid pivot/heatmap and performance-vs-efficiency scatter figures after a batch (or on demand via the "Regenerate" button).
- Output goes to `results/vast/`: `runs_{rma,rmb,rmc}.csv` (one row per run, including a free-text `catatan` column recording *why* that config was tried, plus computed `delta_vs_best_f1_macro_pp`/`is_tie_with_best`/`overfit_signal`), `best.json` (best-so-far per scenario by val F1-macro), `history/` (per-epoch val curves, now including `val_f1_judi`), `checkpoints/`, `metrics/`, `tuning_summary.json` (campaign summary).
- **Hyperparameters are selected on validation only.** `eval_test` defaults to OFF; the test set is opened once, in the **Final** tab.
- The **Final** tab uses each scenario's best config to produce test metrics plus an inference benchmark for RM-a/b/c measured in **one GPU session**, and auto-evaluates the success criteria.

**Chapter 4 validity rule:** efficiency figures (training time, latency, peak memory) are only valid when measured on **one hardware / one session** — never mix Colab, local, and Vast.ai numbers in an efficiency table. F1 is hardware-independent and may be compared across machines.

## Hyperparameters

Starting defaults (`RMA_DEFAULT` / `RMB_DEFAULT` / `RMC_DEFAULT` in `src/tuning.py`) — these are the *entry point* of tuning, not final values:

| Parameter | RM-a | RM-b | RM-c |
|-----------|------|------|------|
| Epochs | 5 | 5 | — |
| Learning rate | 2e-5 | 2e-4 | — |
| Batch size | 16 | 32 | — |
| Warmup ratio | 0.1 | — | — |
| Weight decay | 0.01 | 0.01 | — |
| Max token length | 128 | 128 | 128 |
| Head architecture | — | linear (or `mlp`, `hidden_dim` 256) | inherits best RM-b head |
| Dropout | — | 0.1 | — |
| Alpha (α) | — | — | 0.3 |
| k (neighbors) | — | — | 5 |
| Optimizer | AdamW | AdamW | — (no training) |
| Scheduler | Linear warmup | Linear warmup | — |

Alpha and k in RM-c are tunable — document final values in `DATASET.md` after tuning.

## RM-a hyperparameter exploration design

**Hybrid method** — a combinatorial grid for coupled axes, coordinate descent for independent ones. The partition is derived from the actual mechanics in `src/tuning.py:108-114` (`steps = ceil(N/batch) × epochs`; warmup is a **ratio** of total steps, so it self-normalizes; AdamW weight decay is decoupled by design):

- **Group A — strongly coupled → combinatorial grid:** `lr {1e-5, 2e-5, 3e-5, 5e-5} × epochs {3, 5, 8} × batch {16, 32}` = **24 runs**. These three jointly define the optimization trajectory: batch 32 @ 5 epochs yields 1,030 optimizer updates — exactly half of batch 16 @ 5 epochs (2,060) — so changing one shifts the optimum of the others.
- **Group B — effectively independent → coordinate descent:** `warmup_ratio {0.1 → 0.0}` and `weight_decay {0.01 → 0.1}` = **2 runs** on the winning grid cell.

Total **26 runs ≈ 1.6–1.8 h ≈ $0.40**. A full 5-axis grid (96 runs, ~7 h) was rejected: it consumes nearly the whole budget and inflates the risk of **overfitting to the validation set** (~1,400 samples — the more configs compared, the likelier the winner won by luck).

Run #1 is the canonical IndoNLU/Wilie (2020) baseline (`lr 2e-5 / epochs 5 / batch 16 / warmup 0.1 / wd 0.01`), run first as a sanity check and as the reference point for the whole grid.

**Selection rule:** `val_f1_macro` primary, tie-broken by `val_f1_judi` (class-1 F1 — the minority gambling class, which macro-averaging can mask). Differences ≤ 0.1–0.2 pp count as a tie (single seed 42) → prefer the cheaper config. Read the grid as a *surface*, not a list: pivot `lr × epochs` per `batch` to see whether the optimal lr actually shifts with batch, and treat a winner sitting on the grid edge as a signal to widen the range rather than to lock in.

Full 24-cell table with the rationale for every combination, the Stage-2 runs, the `catatan` template, and the selection rules: **`tuning_grids/RMA_TUNING_GRID.md`** — this is the working document to follow while running the campaign (analogous documents for the other two scenarios: `tuning_grids/RMB_TUNING_GRID.md`, `tuning_grids/RMC_TUNING_GRID.md`; all with matching `.csv` files per stage, ready to upload via Batch mode). Literature basis: Devlin et al. (2019), Wilie et al. (2020, IndoNLU), Sun et al. (2019); full list in `DAFTAR_REFERENSI.pdf`.

## Results Convention

Tuning on Vast.ai writes to `results/vast/`:
- `runs_{rma,rmb,rmc}.csv` — one row per run (config + val metrics + `catatan`), accumulating across sessions
- `best.json` — best-so-far config per scenario, by val F1-macro
- `history/{rma,rmb}_run{id}.csv` — per-epoch val curves (use these to spot overfitting)
- `checkpoints/`, `figures/`, `metrics/` — best weights, confusion matrices, and Final-tab outputs (`inference_benchmark.csv`, `final_comparison.csv`, `success_criteria.csv`)

`run_local_training.py` writes the same kinds of artifacts to `results/local/`.

**Current state of `results/`:** the full Vast.ai tuning campaign + final benchmark are complete (2026-07-25) — see `PROGRESS.md` for the authoritative up-to-date status. `results/vast/` is fully populated: `best.json`, `runs_{rma,rmb,rmc}.csv` (124 runs total), `history/` (per-epoch val curves), `checkpoints/` (`rma_best.pt`, `rmb_best.pt`, `rmc_best.pt`), `figures/` (~70 PNGs), `metrics/` (`final_comparison.csv`, `inference_benchmark.csv`, `success_criteria.csv`, grid pivots), `tuning_summary.json`, `hardware.json`. `results/local/` (from `run_local_training.py`) is still empty — not re-run since the clean-slate. `results/figures/` still holds only the 5 EDA figures (that folder was never for model results).

## Evaluation Metrics

**Classification:** Accuracy, Precision, Recall, F1-score (macro & weighted), confusion matrix.

**Efficiency:** Trainable parameter count, training time (seconds/epoch and total), peak GPU memory (MB), inference latency (ms/sample).

**Success criteria** — a lightweight strategy (RM-b or RM-c) is considered competitive if it meets at least 2 of 3:
- F1-score gap ≤ 3 percentage points vs. RM-a
- Trainable parameter reduction ≥ 90%
- Training time reduction ≥ 50%

## Dataset

`dataset/raw/data_labeling.csv` — labeled YouTube comments (Indonesian). Key columns: `textOriginal` (raw comment text), `label` (0 = normal, 1 = gambling promotion). All other columns (commentId, author, timestamps, etc.) are metadata and can be dropped during preprocessing.

`dataset/processed/` and `dataset/splits/` are **already populated** by `02_preprocessing.ipynb`: 9,395 rows split 70:15:15. Key facts for any modeling code:

- The model input column is **`text_clean`**, not `textOriginal`.
- Class weights live in `dataset/processed/metadata.json` (~`{0: 0.61, 1: 2.75}`) and are passed to `CrossEntropyLoss` — the data is imbalanced, so report macro F1 and class-1 F1, never accuracy alone.
- Preprocessing introduces the special tokens `[URL]`, `[MENTION]`, `[NUM]`, so any model **must** call `resize_token_embeddings` after loading the tokenizer (already handled in `src/modeling.py`).

## Current Status

Data collection, labeling, EDA, preprocessing, and **all tuning are complete** (last updated 2026-07-25 — see `PROGRESS.md` for the authoritative status doc). The full hybrid-design tuning campaigns ran on Vast.ai: RM-a 26 runs, RM-b 31 runs, RM-c 67 runs, plus the Final-tab single-session (RTX 3090) test benchmark. Final test-set results:

| Model | Config | F1-macro | Trainable params | Train time | Inference latency |
|-------|--------|----------|--------------------|------------|--------------------|
| RM-a (full FT) | lr=2e-5, epochs=5, batch=32, warmup=0.1, wd=0.01 | 0.9607 | 109,485,314 | 81.1s | 9.40ms |
| RM-b (frozen + MLP head) | hidden_dim=1024, lr=1e-3, epochs=10, dropout=0.1, wd=0.0, batch=32 | 0.9486 | 789,506 (0.72%) | 11.65s | 9.08ms |
| RM-c (RAC) | alpha=0.2, k=5, weighting=similarity (head = RM-b above) | 0.9497 | 0 | 0.0s | 10.96ms |

RM-b and RM-c both pass 3/3 success criteria. A known limitation: inference latency for RM-b/RM-c is nearly identical to RM-a, since the full 110M-parameter encoder still runs a forward pass at inference in all three scenarios — efficiency gains are in trainable params and training time, not prediction speed.

Remaining work:
1. **Thesis Chapter 4 (Bab 4) write-up** — the main remaining task; use `PROGRESS.md` and `results/vast/` as the source of truth.
2. **(Optional) further RM-c exploration** — e.g. re-checking α/k, or a lighter IndoBERT encoder variant (see `PROGRESS.md`) to attack the inference-latency plateau above. Any such exploration must use a separate `out_dir` (never `results/vast/`, which holds the locked-in Chapter 4 numbers).
