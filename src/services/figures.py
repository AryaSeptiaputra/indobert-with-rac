"""Gambar Bab 4, dibangkitkan langsung dari berkas log kampanye.

Satu builder membaca `runs_*.csv`, `best.json`, dan `metrics/` di folder keluaran
kampanye lalu menulis Gambar 4.1 sampai 4.8 ke `artifacts/gambar/` sebagai PNG 300 dpi
dan PDF vektor. Tidak ada angka yang diketik manual: setiap nilai yang tampil di
gambar berasal dari berkas-berkas itu.

Gaya mengikuti aturan gambar skripsi: Liberation Serif, lebar 5,5 inci, tanpa judul
di dalam gambar, satu keluarga warna biru-nila yang tetap terbaca bila dicetak
hitam putih, jingga hanya untuk penanda konfigurasi final, dan koma sebagai
pemisah desimal.

Pembulatan bawaan Python (dan f-string) memakai representasi biner float,
sehingga 0,125 bisa menjadi 0,12. Anotasi gambar memakai `Decimal` dengan
`ROUND_HALF_UP` supaya angka di gambar sama dengan pembulatan manual di tabel.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src.config import CLASS_NAMES, settings  # noqa: E402
from src.services.selection import rac_per_head, rmc_grid  # noqa: E402
from src.utils.io import read_csv, read_json  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

FIGURE_WIDTH_IN = 5.5
FIGURE_DPI = 300
PP = 100.0

# Gradasi biru-nila terang ke gelap. Luminansinya menurun monoton sehingga keenam
# tingkat tetap terbedakan pada cetakan hitam putih.
BLUE_RAMP = ("#dde5f4", "#b2c3e5", "#7f99d0", "#5270b8", "#324c96", "#1c2b63")
FINAL_ORANGE = "#e07b1a"
WARM_GRAY = "#8a7d70"
NEUTRAL = "#f4f3f1"
INK = "#222222"
MUTED_INK = "#666666"

LABEL_BOX = {"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none"}

SCENARIO_LABELS = {"rma": "RM-a", "rmb": "RM-b", "rmc": "RM-c"}
SCENARIO_COLORS = {"RM-a": BLUE_RAMP[5], "RM-b": BLUE_RAMP[3], "RM-c": BLUE_RAMP[1]}
SCENARIO_MARKERS = {"RM-a": "o", "RM-b": "s", "RM-c": "D"}

RMA_GRID_BATCH = "rma_tahap1_grid"
RMB_CAPACITY_BATCHES = ("rmb_tuning_grid", "rmb_tuning_grid_stage1b")
RMC_GRID_WEIGHTING = "similarity"
LATENCY_COMPONENTS = ("encoder", "classification head", "retrieval FAISS", "fusi")

STYLE = {
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED_INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "axes.grid": False,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.formatter.use_locale": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Serif",
    "mathtext.it": "Liberation Serif:italic",
    "mathtext.bf": "Liberation Serif:bold",
}


def round_half_up(value: float, decimals: int) -> Decimal:
    """Bulatkan setengah ke atas berdasarkan representasi desimal terpendek float."""
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def format_number(value: float, decimals: int = 2, sign: bool = False) -> str:
    """Format gaya Indonesia: `1.234,57`; `sign=True` memberi `+` pada nilai positif.

    Nilai yang dibulatkan menjadi nol ditulis tanpa tanda minus.
    """
    rounded = round_half_up(value, decimals)
    if rounded == 0:
        rounded = abs(rounded)
    text = f"{rounded:,.{decimals}f}"
    if sign and rounded > 0:
        text = "+" + text
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def format_scientific(value: float) -> str:
    """Format learning rate sebagai mathtext `2 × 10^-5` dengan koma desimal."""
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    digits = format_number(mantissa, 0 if np.isclose(mantissa, round(mantissa)) else 1)
    return rf"${digits.replace(',', '{,}')} \times 10^{{{exponent}}}$"


def number_formatter(decimals: int) -> FuncFormatter:
    """Formatter sumbu matplotlib dengan koma desimal."""
    return FuncFormatter(lambda value, _: format_number(value, decimals))


def percent_log_formatter() -> FuncFormatter:
    """Formatter sumbu persen logaritmik: `0,001%`, `1%`, `100%`."""

    def label(value: float, _position: int) -> str:
        decimals = max(0, -int(np.floor(np.log10(value)))) if value > 0 else 0
        return f"{format_number(value, decimals)}%"

    return FuncFormatter(label)


def text_color_for(rgba: tuple[float, float, float, float]) -> str:
    """Tinta putih di atas warna gelap, tinta gelap di atas warna terang."""
    red, green, blue = rgba[:3]
    return "white" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5 else INK


class FigureBuilder:
    """Pembangkit Gambar 4.1 sampai 4.8 untuk satu folder keluaran kampanye.

    Args:
        out_dir: Folder keluaran kampanye (`outputs/tuning/`); `None` memakai
            `settings.default_out_dir`.
    """

    def __init__(self, out_dir: str | Path | None = None) -> None:
        self.out_dir = settings.resolve_out_dir(out_dir)
        self.figures_dir = self.out_dir / "artifacts" / "gambar"

    def build_all(self, skip_missing: bool = True) -> dict[str, list[Path] | str]:
        """Bangun seluruh gambar.

        Args:
            skip_missing: Lewati gambar yang berkas lognya belum ada (misalnya
                Gambar 4.5 sampai 4.8 sebelum `05_final_benchmark.ipynb`
                dijalankan) alih-alih menghentikan seluruh proses.

        Returns:
            Peta nomor gambar ke daftar berkas yang ditulis, atau alasan gambar
            itu dilewati.

        Raises:
            FileNotFoundError: Kalau `skip_missing=False` dan ada log yang hilang.
        """
        builders: dict[str, Callable[[], list[Path]]] = {
            "4.1": self.figure_4_1,
            "4.2": self.figure_4_2,
            "4.3": self.figure_4_3,
            "4.4": self.figure_4_4,
            "4.5": self.figure_4_5,
            "4.6": self.figure_4_6,
            "4.7": self.figure_4_7,
            "4.8": self.figure_4_8,
        }
        results: dict[str, list[Path] | str] = {}
        for number, build in builders.items():
            try:
                results[number] = build()
            except FileNotFoundError as exc:
                if not skip_missing:
                    raise
                logger.warning("Gambar %s dilewati: %s", number, exc)
                results[number] = f"dilewati: {exc}"
        return results

    # ------------------------------------------------------------------
    # Gambar
    # ------------------------------------------------------------------

    def figure_4_1(self) -> list[Path]:
        """Heatmap F1-macro validation kisi RM-a, satu panel per batch."""
        runs = self._runs("rma")
        grid = runs[runs["batch_id"] == RMA_GRID_BATCH]
        if grid.empty:
            raise FileNotFoundError(f"runs_rma.csv belum memuat batch {RMA_GRID_BATCH!r}")

        final = self._best()["rma"]["config"]
        batches = sorted(grid["batch"].unique())
        low, high = grid["val_f1_macro"].min(), grid["val_f1_macro"].max()
        cmap = LinearSegmentedColormap.from_list("biru_nila", BLUE_RAMP)

        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(
                1, len(batches), figsize=(FIGURE_WIDTH_IN, 2.6), squeeze=False,
                constrained_layout=True,
            )
            for ax, batch in zip(axes[0], batches, strict=True):
                pivot = grid[grid["batch"] == batch].pivot_table(
                    index="lr", columns="epochs", values="val_f1_macro", aggfunc="max"
                )
                image = ax.imshow(pivot.values, cmap=cmap, vmin=low, vmax=high, aspect="auto")
                self._annotate_cells(ax, image, pivot.values, lambda v: format_number(v, 4))
                ax.set_xticks(range(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
                ax.set_yticks(range(len(pivot.index)), [format_scientific(r) for r in pivot.index])
                ax.set_xlabel("epoch")
                ax.set_ylabel("learning rate" if ax is axes[0][0] else "")
                ax.set_title(f"batch {int(batch)}")
                self._clear_spines(ax)
                if int(final["batch"]) == int(batch):
                    self._frame_value(ax, pivot, final["lr"], final["epochs"])

            colorbar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.05, pad=0.02)
            colorbar.set_label(
                f"F1-macro validation (rentang {format_number((high - low) * PP, 2)} pp)"
            )
            colorbar.ax.yaxis.set_major_formatter(number_formatter(3))
            return self._save(fig, "gambar_4_1")

    def figure_4_2(self) -> list[Path]:
        """Kurva kapasitas head RM-b: F1-macro validation terhadap trainable params."""
        runs = self._runs("rmb")
        points = (
            runs[runs["batch_id"].isin(RMB_CAPACITY_BATCHES)]
            .sort_values("trainable_params")
            .reset_index(drop=True)
        )
        if len(points) < 2:
            raise FileNotFoundError(
                f"runs_rmb.csv belum memuat batch kapasitas {RMB_CAPACITY_BATCHES}"
            )

        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 2.8), constrained_layout=True)
            ax.plot(
                points["trainable_params"], points["val_f1_macro"],
                color=BLUE_RAMP[4], linewidth=1.5, marker="o", markersize=5,
            )
            for i, row in points.iterrows():
                ax.annotate(
                    self._head_label(row), (row["trainable_params"], row["val_f1_macro"]),
                    textcoords="offset points", xytext=(6, -11), ha="left",
                    fontsize=7.5, color=MUTED_INK, bbox=LABEL_BOX,
                )
                if i == 0:
                    continue
                before = points.loc[i - 1]
                gain = (row["val_f1_macro"] - before["val_f1_macro"]) * PP
                per_100k = gain / (row["trainable_params"] - before["trainable_params"]) * 1e5
                ax.annotate(
                    format_number(per_100k, 2, sign=True),
                    (
                        (row["trainable_params"] + before["trainable_params"]) / 2,
                        (row["val_f1_macro"] + before["val_f1_macro"]) / 2,
                    ),
                    textcoords="offset points", xytext=(-4, 5), ha="right",
                    fontsize=7.5, color=INK, bbox=LABEL_BOX,
                )

            ax.text(
                0.99, 0.03, "angka pada ruas: kenaikan marjinal, pp per 100.000 parameter",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED_INK,
            )
            ax.set_xlabel("trainable parameters")
            ax.set_ylabel("F1-macro validation")
            ax.xaxis.set_major_formatter(number_formatter(0))
            ax.yaxis.set_major_formatter(number_formatter(3))
            ax.grid(axis="y")
            margin = (points["val_f1_macro"].max() - points["val_f1_macro"].min()) * 0.25
            ax.set_ylim(points["val_f1_macro"].min() - margin, points["val_f1_macro"].max() + margin)
            return self._save(fig, "gambar_4_2")

    def figure_4_3(self) -> list[Path]:
        """Heatmap divergen selisih RAC terhadap tanpa RAC, pada head juara RM-c."""
        final = self._best()["rmc"]["config"]
        head_id = int(final["rmb_run_id"])
        grid = self._rmc_grid()
        grid = grid[grid["rmb_run_id"] == head_id]
        pivot = grid.pivot_table(index="alpha", columns="k", values="gain_pp", aggfunc="max")

        limit = max(float(np.nanmax(np.abs(pivot.values))), settings.tie_threshold_pp * 2)
        cmap = LinearSegmentedColormap.from_list(
            "divergen", [WARM_GRAY, NEUTRAL, BLUE_RAMP[2], BLUE_RAMP[5]]
        )
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 4.2), constrained_layout=True)
            image = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
            self._annotate_cells(ax, image, pivot.values, lambda v: format_number(v, 2, sign=True))

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    if abs(pivot.values[i, j]) <= settings.tie_threshold_pp + 1e-9:
                        ax.add_patch(Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                            edgecolor="#b5b5b5", linewidth=0,
                        ))
            self._frame_value(ax, pivot, final["alpha"], final["k"])

            ax.set_xticks(range(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [format_number(a, 1) for a in pivot.index])
            ax.set_xlabel("k (jumlah tetangga)")
            ax.set_ylabel("α (bobot cabang retrieval)")
            self._clear_spines(ax)

            colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
            colorbar.set_label("selisih F1-macro validation terhadap tanpa RAC (pp)")
            colorbar.ax.yaxis.set_major_formatter(number_formatter(2))
            ax.legend(
                handles=[
                    Patch(facecolor="white", edgecolor="#b5b5b5", hatch="////",
                          label=f"di dalam ambang seri ±{format_number(settings.tie_threshold_pp, 2)} pp"),
                    Patch(facecolor="none", edgecolor=FINAL_ORANGE, linewidth=1.8,
                          label="konfigurasi final"),
                ],
                loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
            )
            return self._save(fig, "gambar_4_3")

    def figure_4_4(self) -> list[Path]:
        """Dumbbell kontribusi RAC per head RM-b: tanpa RAC, terbaik, dan bersama."""
        summary = self.rac_per_head()
        final = self._best()["rmc"]
        official_head = int(self._best()["rmb"]["run_id"])
        final_head = int(final["config"]["rmb_run_id"])

        rows = summary.sort_values("f1_no_rac", ascending=True).reset_index(drop=True)
        positions = np.arange(len(rows))

        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 5.0), constrained_layout=True)
            ax.hlines(positions, rows["f1_no_rac"], rows["f1_best"],
                      color=BLUE_RAMP[2], linewidth=1.2, zorder=1)
            ax.scatter(rows["f1_no_rac"], positions, s=22, color=BLUE_RAMP[1],
                       edgecolor=BLUE_RAMP[4], linewidth=0.6, zorder=2, label="tanpa RAC")
            ax.scatter(rows["f1_shared"], positions, s=26, facecolor="white",
                       edgecolor=BLUE_RAMP[5], linewidth=1.0, zorder=3,
                       label="konfigurasi fusi bersama")
            ax.scatter(rows["f1_best"], positions, s=26, color=BLUE_RAMP[5], zorder=4,
                       label="konfigurasi fusi terbaik per head")

            grid = self._rmc_grid()
            final_cell = grid[
                (grid["rmb_run_id"] == final_head)
                & np.isclose(grid["alpha"], float(final["config"]["alpha"]))
                & (grid["k"] == int(final["config"]["k"]))
            ]
            final_rows = rows[rows["rmb_run_id"] == final_head]
            ax.scatter(final_cell["val_f1_macro"].iloc[:1].tolist() * len(final_rows), final_rows.index, s=70,
                       facecolor="none", edgecolor=FINAL_ORANGE, linewidth=1.8, zorder=5,
                       label="konfigurasi final")

            span = rows[["f1_no_rac", "f1_best"]].to_numpy()
            offset = (span.max() - span.min()) * 0.02
            for position, row in rows.iterrows():
                ax.text(
                    max(row["f1_best"], row["f1_shared"]) + offset, position,
                    format_number(row["gain_best_pp"], 2, sign=True),
                    va="center", fontsize=7, color=INK,
                )

            ax.set_yticks(positions, [self._head_label(row, with_id=True) for _, row in rows.iterrows()])
            for label, (_, row) in zip(ax.get_yticklabels(), rows.iterrows(), strict=True):
                if row["head_arch"] == "linear" or int(row["rmb_run_id"]) == official_head:
                    label.set_fontweight("bold")
                    label.set_color(INK)
            ax.set_ylim(len(rows) - 0.5, -0.5)
            ax.set_xlabel("F1-macro validation")
            ax.xaxis.set_major_formatter(number_formatter(3))
            ax.set_xlim(span.min() - offset * 2, span.max() + offset * 8)
            ax.grid(axis="x")
            ax.tick_params(axis="y", length=0, labelsize=7)
            ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=2)
            return self._save(fig, "gambar_4_4")

    def figure_4_5(self) -> list[Path]:
        """Confusion matrix test ketiga skenario, dinormalisasi per kelas aktual."""
        predictions = self._metrics_csv("final_predictions.csv")
        cmap = LinearSegmentedColormap.from_list("biru_nila", BLUE_RAMP)
        labels = list(CLASS_NAMES)

        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.3), constrained_layout=True)
            for ax, scenario in zip(axes, ("rma", "rmb", "rmc"), strict=True):
                counts = pd.crosstab(
                    predictions["label"], predictions[f"pred_{scenario}"]
                ).reindex(index=range(len(labels)), columns=range(len(labels)), fill_value=0)
                matrix = counts.to_numpy()
                share = matrix / matrix.sum(axis=1, keepdims=True)
                image = ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(
                            j, i,
                            f"{format_number(matrix[i, j], 0)}\n({format_number(share[i, j] * PP, 1)}%)",
                            ha="center", va="center", fontsize=7.5,
                            color=text_color_for(image.cmap(image.norm(share[i, j]))),
                        )
                ax.set_xticks(range(len(labels)), labels)
                ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [])
                ax.set_xlabel("prediksi")
                ax.set_ylabel("aktual" if ax is axes[0] else "")
                ax.set_title(SCENARIO_LABELS[scenario])
                self._clear_spines(ax)

            colorbar = fig.colorbar(image, ax=list(axes), fraction=0.04, pad=0.02)
            colorbar.set_label("proporsi per kelas aktual (%)")
            colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_number(v * PP, 0)))
            return self._save(fig, "gambar_4_5")

    def figure_4_6(self) -> list[Path]:
        """Biaya pelatihan RM-b dan RM-c relatif terhadap RM-a, sumbu log."""
        costs = self.training_costs()
        metrics = (
            ("trainable_params", "trainable parameters"),
            ("train_time_s", "training time"),
            ("train_peak_mem_mb", "memori GPU pelatihan"),
        )
        baseline = costs.loc["RM-a"]
        if (costs[[m for m, _ in metrics]] <= 0).any().any():
            raise ValueError("biaya pelatihan harus positif untuk sumbu logaritmik")

        width = 0.26
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.0), constrained_layout=True)
            for offset, scenario in zip((-width, 0.0, width), costs.index, strict=True):
                relative = [costs.loc[scenario, m] / baseline[m] * PP for m, _ in metrics]
                bars = ax.bar(
                    np.arange(len(metrics)) + offset, relative, width=width * 0.92,
                    color=SCENARIO_COLORS[scenario], label=scenario,
                    hatch="////" if scenario == "RM-c" else None,
                    edgecolor=BLUE_RAMP[4] if scenario == "RM-c" else "white", linewidth=0.5,
                )
                for bar, value in zip(bars, relative, strict=True):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, value * 1.15,
                        f"{format_number(value, self._percent_decimals(value))}%",
                        ha="center", va="bottom", fontsize=7, color=INK,
                    )

            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(percent_log_formatter())
            ax.set_ylim(top=PP * 8)
            ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
            ax.set_ylabel("persentase terhadap RM-a (skala log)")
            ax.grid(axis="y", which="major")
            ax.legend(
                handles=[
                    Patch(facecolor=SCENARIO_COLORS["RM-a"], label="RM-a"),
                    Patch(facecolor=SCENARIO_COLORS["RM-b"], label="RM-b"),
                    Patch(facecolor=SCENARIO_COLORS["RM-c"], hatch="////",
                          edgecolor=BLUE_RAMP[4], label="RM-c (diwarisi dari head RM-b)"),
                ],
                loc="upper right", ncol=3,
            )
            return self._save(fig, "gambar_4_6")

    def figure_4_7(self) -> list[Path]:
        """Stacked bar dekomposisi latency inferensi per skenario."""
        breakdown = self._metrics_csv("latency_breakdown.csv")
        table = (
            breakdown.pivot_table(index="scenario", columns="component", values="latency_ms")
            .reindex(index=["RM-a", "RM-b", "RM-c"], columns=list(LATENCY_COMPONENTS))
            .fillna(0.0)
        )
        colors = (BLUE_RAMP[4], BLUE_RAMP[2], BLUE_RAMP[1], BLUE_RAMP[0])
        hatches = (None, None, "....", "////")

        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.0), constrained_layout=True)
            bottom = np.zeros(len(table))
            for component, color, hatch in zip(LATENCY_COMPONENTS, colors, hatches, strict=True):
                values = table[component].to_numpy()
                ax.bar(table.index, values, bottom=bottom, width=0.5, color=color,
                       hatch=hatch, edgecolor="white", linewidth=0.8, label=component)
                bottom += values

            totals = table.sum(axis=1)
            for position, (scenario, total) in enumerate(totals.items()):
                ax.text(position, total * 1.02, f"{format_number(total, 2)} ms",
                        ha="center", va="bottom", fontsize=7.5, color=INK)
                share = table.loc[scenario, "encoder"] / total * PP
                ax.text(position, table.loc[scenario, "encoder"] / 2,
                        f"encoder\n{format_number(share, 1)}%",
                        ha="center", va="center", fontsize=7, color="white")

            ax.set_ylim(0, totals.max() * 1.15)
            ax.set_ylabel("latency per sampel (ms)")
            ax.yaxis.set_major_formatter(number_formatter(1))
            ax.grid(axis="y")
            ax.legend(loc="upper left", ncol=4)
            return self._save(fig, "gambar_4_7")

    def figure_4_8(self) -> list[Path]:
        """Scatter F1-macro test terhadap training time dan inference latency."""
        comparison = self._metrics_csv("final_comparison.csv").set_index("model")
        threshold = comparison.loc["RM-a", "test_f1_macro"] - 0.03

        with plt.rc_context(STYLE):
            fig, (left, right) = plt.subplots(
                1, 2, figsize=(FIGURE_WIDTH_IN, 2.6), sharey=True, constrained_layout=True
            )
            for ax, column, label in (
                (left, "train_time_s", "training time (s, skala log)"),
                (right, "infer_latency_ms", "inference latency (ms)"),
            ):
                for scenario, row in comparison.iterrows():
                    ax.scatter(row[column], row["test_f1_macro"], s=36,
                               color=SCENARIO_COLORS[scenario], marker=SCENARIO_MARKERS[scenario],
                               edgecolor="white", linewidth=0.6, zorder=3)
                    ax.annotate(scenario, (row[column], row["test_f1_macro"]),
                                textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=INK)
                ax.axhline(threshold, color=MUTED_INK, linestyle="--", linewidth=0.8)
                ax.set_xlabel(label)
                ax.grid(alpha=1.0)
            left.set_xscale("log")
            left.set_xlim(comparison["train_time_s"].min() / 2, comparison["train_time_s"].max() * 3)
            left.xaxis.set_major_formatter(number_formatter(0))
            # Mulai dari nol: pesannya adalah ketiga skenario TIDAK bergeser di fase
            # inferensi, dan sumbu yang dipotong akan membesar-besarkan selisih kecil.
            right.set_xlim(0, comparison["infer_latency_ms"].max() * 1.3)
            right.xaxis.set_major_formatter(number_formatter(1))
            left.yaxis.set_major_formatter(number_formatter(3))
            left.set_ylabel("F1-macro test")
            left.text(0.02, threshold, " ambang 3 pp di bawah RM-a", transform=left.get_yaxis_transform(),
                      va="bottom", fontsize=7, color=MUTED_INK)
            low = min(threshold, comparison["test_f1_macro"].min())
            high = comparison["test_f1_macro"].max()
            left.set_ylim(low - (high - low) * 0.15, high + (high - low) * 0.15)
            return self._save(fig, "gambar_4_8")

    # ------------------------------------------------------------------
    # Tabel turunan (juga dipakai test)
    # ------------------------------------------------------------------

    def rac_per_head(self) -> pd.DataFrame:
        """Ringkasan RAC per head untuk Gambar 4.4 (lihat `rac_summary.rac_per_head`)."""
        return rac_per_head(self._runs("rmc"), self._runs("rmb"))

    def training_costs(self) -> pd.DataFrame:
        """Biaya pelatihan juara tiap skenario untuk Gambar 4.6.

        RM-c tidak melatih apa pun; biayanya adalah biaya head RM-b yang dipakai
        juaranya, diambil dari baris head itu di `runs_rmb.csv`.

        Returns:
            DataFrame berindeks RM-a, RM-b, RM-c dengan kolom `trainable_params`,
            `train_time_s`, dan `train_peak_mem_mb`.
        """
        best = self._best()
        rma = self._runs("rma").set_index("run_id").loc[int(best["rma"]["run_id"])]
        heads = self._runs("rmb").set_index("run_id")
        rmb = heads.loc[int(best["rmb"]["run_id"])]
        rmc = heads.loc[int(best["rmc"]["config"]["rmb_run_id"])]

        def head_cost(row: pd.Series) -> dict[str, float]:
            return {
                "trainable_params": float(row["trainable_params"]),
                "train_time_s": float(row["train_time_s"]),
                "train_peak_mem_mb": float(row["train_peak_mem_mb"]),
            }

        return pd.DataFrame(
            {
                "RM-a": {
                    "trainable_params": float(rma["trainable_params"]),
                    "train_time_s": float(rma["train_time_s"]),
                    "train_peak_mem_mb": float(rma["peak_mem_mb"]),
                },
                "RM-b": head_cost(rmb),
                "RM-c": head_cost(rmc),
            }
        ).T

    # ------------------------------------------------------------------
    # Pembantu
    # ------------------------------------------------------------------

    def _runs(self, scenario: str) -> pd.DataFrame:
        frame = read_csv(self.out_dir / f"runs_{scenario}.csv")
        if frame.empty:
            raise FileNotFoundError(f"runs_{scenario}.csv belum ada atau kosong di {self.out_dir}")
        return frame

    def _best(self) -> dict[str, dict[str, object]]:
        best = read_json(self.out_dir / "best.json", default={}) or {}
        if not best:
            raise FileNotFoundError(f"best.json belum ada di {self.out_dir}")
        return best

    def _metrics_csv(self, filename: str) -> pd.DataFrame:
        frame = read_csv(self.out_dir / "metrics" / filename)
        if frame.empty:
            raise FileNotFoundError(
                f"metrics/{filename} belum ada; jalankan 05_final_benchmark.ipynb lebih dulu"
            )
        return frame

    def _rmc_grid(self) -> pd.DataFrame:
        return rmc_grid(self._runs("rmc"))

    @staticmethod
    def _head_label(row: pd.Series, with_id: bool = False) -> str:
        name = "linear" if row["head_arch"] == "linear" else f"MLP {int(row['hidden_dim'])}"
        return f"#{int(row['rmb_run_id'] if 'rmb_run_id' in row else row['run_id'])} {name}" if with_id else name

    @staticmethod
    def _percent_decimals(value: float) -> int:
        if value >= 10:
            return 0
        return max(1, -int(np.floor(np.log10(value))) + 1)

    @staticmethod
    def _annotate_cells(ax, image, values: np.ndarray, formatter: Callable[[float], str]) -> None:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if np.isnan(value):
                    continue
                ax.text(
                    j, i, formatter(value), ha="center", va="center", fontsize=7,
                    color=text_color_for(image.cmap(image.norm(value))),
                )

    @classmethod
    def _frame_value(cls, ax, pivot: pd.DataFrame, row_value: float, column_value: float) -> None:
        rows = np.flatnonzero(np.isclose(pivot.index.astype(float), float(row_value)))
        columns = np.flatnonzero(np.isclose(pivot.columns.astype(float), float(column_value)))
        if len(rows) and len(columns):
            cls._frame_cell(ax, int(rows[0]), int(columns[0]))

    @staticmethod
    def _frame_cell(ax, row: int, column: int) -> None:
        ax.add_patch(Rectangle(
            (column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=FINAL_ORANGE,
            linewidth=1.8, zorder=5,
        ))

    @staticmethod
    def _clear_spines(ax) -> None:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    def _save(self, fig: plt.Figure, stem: str) -> list[Path]:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        paths = [self.figures_dir / f"{stem}.png", self.figures_dir / f"{stem}.pdf"]
        for path in paths:
            fig.savefig(path, dpi=FIGURE_DPI)
        plt.close(fig)
        logger.info("Gambar ditulis: %s", paths[0].name)
        return paths


__all__ = ["FigureBuilder", "format_number", "format_scientific", "round_half_up"]
