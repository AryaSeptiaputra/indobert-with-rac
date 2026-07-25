"""Metrik klasifikasi & pengukuran efisiensi komputasi (RM-a/b/c).

Metrik performa : accuracy, precision/recall/F1 (macro & weighted) + per-kelas,
                  confusion matrix. Metrik utama = F1-macro.
Metrik efisiensi: jumlah trainable params, waktu training, peak GPU memory,
                  inference latency (ms/sampel).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)

CLASS_NAMES = ["non-judi (0)", "judi (1)"]


def classification_metrics(y_true, y_pred) -> dict:
    """Kembalikan dict metrik performa lengkap (macro, weighted, per-kelas)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    p_c, r_c, f_c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f_mac),
        "precision_macro": float(p_mac),
        "recall_macro": float(r_mac),
        "f1_weighted": float(f_wt),
        "precision_weighted": float(p_wt),
        "recall_weighted": float(r_wt),
        "f1_class0": float(f_c[0]), "f1_class1": float(f_c[1]),
        "precision_class0": float(p_c[0]), "precision_class1": float(p_c[1]),
        "recall_class0": float(r_c[0]), "recall_class1": float(r_c[1]),
        "support_class0": int(sup[0]), "support_class1": int(sup[1]),
    }


def plot_confusion_matrix(y_true, y_pred, path, title="Confusion Matrix"):
    """Simpan figur confusion matrix ke `path`. Pakai backend Agg (headless)."""
    import matplotlib
    matplotlib.use("Agg")  # non-interaktif: simpan ke file tanpa butuh display/Tk
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return cm


def plot_pr_curve(y_true, y_score, path, title="Precision-Recall (kelas judi)"):
    """PR curve kelas positif (1=judi) dari skor probabilitas -> simpan PNG di `path`.

    Relevan untuk data imbalance (4,5:1): lebih informatif dari confusion matrix
    tunggal untuk melihat trade-off precision/recall pada kelas minoritas.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = float(y_true.mean()) if len(y_true) else 0.0
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(rec, prec, color="tab:purple")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
              label=f"baseline ({baseline:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{title}\nAP={ap:.4f}")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return ap


def count_parameters(model) -> dict:
    """Total & trainable parameter (+persentase trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_pct": float(trainable / total * 100) if total else 0.0,
    }


def peak_gpu_mem_mb() -> float:
    """Peak GPU memory teralokasi (MB) sejak reset terakhir. 0 bila CPU."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


@torch.no_grad()
def measure_latency(predict_fn, sample, n_warmup: int = 5, n_runs: int = 50) -> float:
    """Rata-rata latency inferensi (ms/sampel) untuk `predict_fn(sample)`.

    predict_fn harus melakukan satu forward untuk satu sampel. Sinkronisasi CUDA
    diperhitungkan bila di GPU.
    """
    cuda = torch.cuda.is_available()
    for _ in range(n_warmup):
        predict_fn(sample)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(sample)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000.0


def save_metrics_csv(metrics: dict, path):
    """Tulis satu-baris CSV (header = keys). Buat folder bila perlu."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(metrics.keys()))
        w.writerow([metrics[k] for k in metrics])
    return path
