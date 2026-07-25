"""Artefak turunan pasca-run: pivot 'grid sebagai permukaan', heatmap, scatter
F1-vs-efisiensi, dan ringkasan campaign (tuning_summary.json).

Tidak bergantung pada torch -- bisa dipanggil langsung dari proses Streamlit (tombol
manual di app.py) maupun dari subprocess job_runner.py di akhir tiap job/batch.

Semua fungsi BEST-EFFORT: butuh >=2 nilai unik di >=2 kolom HP untuk pivot/heatmap, dan
>=2 baris untuk scatter -- kembalikan [] / None secara diam-diam bila prasyarat tak
terpenuhi (mis. baru ada segelintir run ad-hoc), tidak pernah melempar exception ke
pemanggil untuk kondisi data yang memang belum cukup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# (baris, kolom, facet) per skenario -- sesuai sumbu yang dibahas RMA_TUNING_GRID.md
# ("baca grid sebagai permukaan": pivot baris x kolom, terpisah per nilai facet).
AXES = {
    "rma": ("lr", "epochs", "batch"),
    "rmb": ("lr", "epochs", "hidden_dim"),
    "rmc": ("alpha", "k", "weighting"),
}


def grid_pivot_and_heatmap(out: Path, scenario: str, metric: str = "val_f1_macro") -> list[Path]:
    """Pivot `metric` (baris x kolom, satu pivot per nilai facet) + heatmap PNG.

    Best-effort: return [] bila file run belum ada, kolom sumbu tak lengkap, atau
    variasi nilai belum cukup (<2 nilai unik di baris/kolom) untuk membentuk grid nyata.
    """
    out = Path(out)
    csv = out / f"runs_{scenario}.csv"
    if not csv.exists():
        return []
    d = pd.read_csv(csv)
    row_f, col_f, facet_f = AXES.get(scenario, (None, None, None))
    if not row_f or not all(c in d.columns for c in (row_f, col_f, facet_f, metric)):
        return []
    d = d.dropna(subset=[row_f, col_f, facet_f, metric])
    if d[row_f].nunique() < 2 or d[col_f].nunique() < 2:
        return []

    written: list[Path] = []
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for facet_val, sub in d.groupby(facet_f):
        piv = sub.pivot_table(index=row_f, columns=col_f, values=metric, aggfunc="max")
        csv_p = out / "metrics" / f"{scenario}_grid_pivot_{facet_f}{facet_val}.csv"
        piv.to_csv(csv_p)
        written.append(csv_p)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(1.2 * len(piv.columns) + 2, 1.0 * len(piv.index) + 2))
        im = ax.imshow(piv.values, cmap="viridis")
        ax.set_xticks(range(len(piv.columns)), labels=[str(c) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), labels=[str(r) for r in piv.index])
        ax.set_xlabel(col_f); ax.set_ylabel(row_f)
        ax.set_title(f"{scenario.upper()} {metric} -- {facet_f}={facet_val}")
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        png_p = out / "figures" / f"{scenario}_grid_heatmap_{facet_f}{facet_val}.png"
        fig.savefig(png_p, dpi=120)
        plt.close(fig)
        written.append(png_p)
    return written


def tradeoff_scatter(out: Path, scenario: str, x: str = "train_time_s", y: str = "val_f1_macro",
                     size_by: str = "trainable_params", color_by: str = "peak_mem_mb") -> Path | None:
    """Scatter performa (y) vs efisiensi (x) DI DALAM satu skenario -- pelengkap
    `metrics/final_comparison.csv` yang hanya membandingkan LINTAS skenario di tab Final.

    `size_by` (jumlah trainable params) dan `color_by` (peak GPU memory) menambah 2
    dimensi efisiensi lagi tanpa chart terpisah -- keduanya best-effort, diabaikan diam-diam
    kalau kolomnya tak ada/kosong (mis. RM-c tak punya peak_mem_mb).
    """
    out = Path(out)
    csv = out / f"runs_{scenario}.csv"
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    if len(d) < 2 or x not in d.columns or y not in d.columns:
        return None
    d = d.dropna(subset=[x, y])
    if len(d) < 2:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = None
    if size_by in d.columns and d[size_by].notna().any():
        s = d[size_by].fillna(d[size_by].median())
        rng = s.max() - s.min()
        sizes = 30 + 200 * (s - s.min()) / (rng + 1e-9) if rng > 0 else None

    colors = None
    if color_by in d.columns and d[color_by].notna().any() and d[color_by].nunique() > 1:
        colors = d[color_by]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(d[x], d[y], s=sizes if sizes is not None else 60,
                    c=colors if colors is not None else "tab:blue",
                    cmap="plasma" if colors is not None else None, alpha=0.8, edgecolors="black")
    if colors is not None:
        fig.colorbar(sc, ax=ax, label=color_by)
    for _, r in d.iterrows():
        if "run_id" in d.columns:
            ax.annotate(str(int(r["run_id"])), (r[x], r[y]), fontsize=7, alpha=0.7)
    ax.set_xlabel(x); ax.set_ylabel(y)
    title = f"{scenario.upper()} -- {y} vs {x} (ukuran = {size_by}"
    title += f", warna = {color_by})" if colors is not None else ")"
    ax.set_title(title)
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    p = out / "figures" / f"{scenario}_tradeoff.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def training_curve(out: Path, scenario: str, run_id: int, hist: list) -> Path | None:
    """2-panel: train_loss vs epoch (kiri) + val_f1_macro & val_f1_judi vs epoch (kanan)
    -> figures/{scenario}_run{id}_curve.png. Murah (cuma plot hist yang sudah dihitung,
    tanpa forward pass baru) -- dipanggil untuk SETIAP run, bukan cuma juara.

    Best-effort: None kalau hist kosong atau kolom inti (epoch/val_f1_macro) tak ada.
    """
    out = Path(out)
    if not hist:
        return None
    d = pd.DataFrame(hist)
    if "epoch" not in d.columns or "val_f1_macro" not in d.columns:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if "train_loss" in d.columns:
        axes[0].plot(d["epoch"], d["train_loss"], marker="o", color="tab:red")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train_loss")
        axes[0].set_title("Training loss")
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "train_loss tak tersedia", ha="center", va="center")

    axes[1].plot(d["epoch"], d["val_f1_macro"], marker="o", label="val F1-macro", color="tab:blue")
    if "val_f1_judi" in d.columns:
        axes[1].plot(d["epoch"], d["val_f1_judi"], marker="s", label="val F1-judi", color="tab:orange")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("F1"); axes[1].legend()
    axes[1].set_title("Validation F1")

    fig.suptitle(f"{scenario.upper()} run #{run_id}")
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    p = out / "figures" / f"{scenario}_run{run_id}_curve.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def top_configs_bar_chart(out: Path, scenario: str, metric: str = "val_f1_macro",
                          top_n: int = 10) -> Path | None:
    """Horizontal bar chart top-N run terurut oleh `metric` -- lebih cepat dibaca
    daripada tabel datar 24+ baris. Label = run_id + 2 field HP paling informatif
    (AXES[scenario][:2]) -> figures/{scenario}_top_configs.png.
    """
    out = Path(out)
    csv = out / f"runs_{scenario}.csv"
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    if metric not in d.columns or len(d) == 0:
        return None
    d = d.dropna(subset=[metric])
    if len(d) == 0:
        return None

    label_fields = [f for f in AXES.get(scenario, ())[:2] if f in d.columns]
    top = d.nlargest(min(top_n, len(d)), metric).sort_values(metric)

    def _label(r):
        parts = [f"{f}={r[f]:g}" if isinstance(r[f], (int, float)) else f"{f}={r[f]}"
                for f in label_fields if pd.notna(r[f])]
        rid = int(r["run_id"]) if "run_id" in d.columns and pd.notna(r["run_id"]) else "?"
        return f"#{rid} " + " ".join(parts)

    labels = [_label(r) for _, r in top.iterrows()]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(top) + 1.5))
    ax.barh(range(len(top)), top[metric], color="tab:green")
    ax.set_yticks(range(len(top)), labels=labels, fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(f"{scenario.upper()} -- top {len(top)} konfigurasi by {metric}")
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    p = out / "figures" / f"{scenario}_top_configs.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def final_inference_bar_chart(out: Path) -> Path | None:
    """2-panel bar chart RM-a/b/c dari metrics/inference_benchmark.csv (tab Final, satu
    sesi GPU): latency (kiri) + peak GPU memory inferensi (kanan). Pelengkap visual untuk
    tabel `inference_benchmark.csv` yang selama ini murni angka.
    """
    out = Path(out)
    csv = out / "metrics" / "inference_benchmark.csv"
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    if "scenario" not in d.columns or len(d) == 0:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    if "infer_latency_ms" in d.columns:
        axes[0].bar(d["scenario"], d["infer_latency_ms"], color=colors[:len(d)])
        axes[0].set_ylabel("ms/sampel"); axes[0].set_title("Latency inferensi")
    else:
        axes[0].axis("off")
    if "infer_peak_gpu_mem_mb" in d.columns:
        axes[1].bar(d["scenario"], d["infer_peak_gpu_mem_mb"], color=colors[:len(d)])
        axes[1].set_ylabel("MB"); axes[1].set_title("Peak GPU memory inferensi")
    else:
        axes[1].axis("off")
    fig.suptitle("Benchmark inferensi RM-a/b/c (satu sesi GPU)")
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    p = out / "figures" / "final_inference_benchmark.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def write_run_bundle_summary(out: Path) -> Path:
    """tuning_summary.json -- jumlah run & rentang waktu tiap skenario + juara sejauh ini.

    Mengisi janji `VAST_GUIDE.md` yang sebelumnya tidak ditulis kode manapun.
    `config_used.json` SENGAJA tidak dibuat sebagai file terpisah -- best.json sudah
    berisi persis itu per skenario (lihat VAST_GUIDE.md yang sudah diperbarui).
    """
    out = Path(out)
    best_p = out / "best.json"
    best = json.load(open(best_p, encoding="utf-8")) if best_p.exists() else {}
    summary = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "scenarios": {}}
    for sc in ["rma", "rmb", "rmc"]:
        p = out / f"runs_{sc}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        entry = {"n_runs": len(d)}
        if "timestamp" in d.columns and len(d):
            entry["first_run_at"] = str(d["timestamp"].min())
            entry["last_run_at"] = str(d["timestamp"].max())
        if "batch_id" in d.columns:
            n_batches = d["batch_id"].dropna().astype(str).replace("", np.nan).dropna().nunique()
            entry["n_batches"] = int(n_batches)
        if sc in best:
            entry["best_run_id"] = best[sc].get("run_id")
            entry["best_val_f1_macro"] = best[sc].get("val_f1_macro")
        summary["scenarios"][sc] = entry
    p = out / "tuning_summary.json"
    json.dump(summary, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return p
