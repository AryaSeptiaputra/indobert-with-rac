"""Artefak visual dan ringkasan turunan dari riwayat run.

Menghasilkan kurva training per-run, confusion matrix, PR curve, pivot dan
heatmap grid, scatter performa-versus-efisiensi, bar chart konfigurasi teratas,
serta `tuning_summary.json`.

Backend Matplotlib dikunci ke Agg sekali di level modul. Versi sebelumnya
menyisipkan blok `matplotlib.use("Agg")` beserta impornya lima kali, salah
satunya di dalam loop `groupby` sehingga terpanggil ulang untuk tiap facet.

Sebagian besar method bersifat best-effort: pivot dan heatmap butuh minimal dua
nilai unik pada dua sumbu, scatter butuh minimal dua baris. Prasyarat yang belum
terpenuhi mengembalikan `None` atau daftar kosong disertai log, bukan exception,
karena kondisi itu wajar di awal kampanye saat baru ada segelintir run.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from src.config import CLASS_NAMES, SCENARIOS  # noqa: E402
from src.services.evaluation import ClassificationEvaluator  # noqa: E402
from src.utils.io import read_csv, read_json, write_csv, write_json  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

FIGURE_DPI = 120

# (baris, kolom, facet) tiap skenario, mengikuti sumbu yang dibahas di
# tuning_grids/*.md: grid dibaca sebagai permukaan, bukan daftar datar.
GRID_AXES: dict[str, tuple[str, str, str]] = {
    "rma": ("lr", "epochs", "batch"),
    "rmb": ("lr", "epochs", "hidden_dim"),
    "rmc": ("alpha", "k", "weighting"),
}


class FigureReporter:
    """Pembuat gambar dan ringkasan untuk satu folder keluaran kampanye.

    Args:
        out_dir: Folder keluaran kampanye; seluruh artefak ditulis relatif ke sini.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.evaluator = ClassificationEvaluator()

    @property
    def figures_dir(self) -> Path:
        """Folder gambar, dibuat bila belum ada."""
        path = self.out_dir / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def metrics_dir(self) -> Path:
        """Folder tabel metrik, dibuat bila belum ada."""
        path = self.out_dir / "metrics"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def runs_frame(self, scenario: str) -> pd.DataFrame:
        """Baca riwayat run satu skenario.

        Args:
            scenario: Kode skenario.

        Returns:
            DataFrame riwayat; kosong bila berkasnya belum ada.
        """
        return read_csv(self.out_dir / f"runs_{scenario}.csv")

    def training_curve(
        self,
        scenario: str,
        run_id: int,
        history: list[dict[str, float | int]],
    ) -> Path | None:
        """Kurva loss training dan F1 validation per epoch untuk satu run.

        Murah karena hanya memplot angka yang sudah dihitung, tanpa forward pass
        baru, sehingga dibuat untuk SETIAP run dan bukan hanya untuk juaranya.

        Args:
            scenario: Kode skenario.
            run_id: Nomor run.
            history: Baris per-epoch dari hasil training.

        Returns:
            Path PNG, atau `None` bila history tidak memuat kolom inti.
        """
        frame = pd.DataFrame(history)
        if frame.empty or not {"epoch", "val_f1_macro"} <= set(frame.columns):
            logger.debug("Kurva %s run %d dilewati: history tidak lengkap", scenario, run_id)
            return None

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        if "train_loss" in frame.columns:
            axes[0].plot(frame["epoch"], frame["train_loss"], marker="o", color="tab:red")
            axes[0].set_xlabel("epoch")
            axes[0].set_ylabel("train_loss")
            axes[0].set_title("Training loss")
        else:
            axes[0].axis("off")
            axes[0].text(0.5, 0.5, "train_loss tak tersedia", ha="center", va="center")

        axes[1].plot(
            frame["epoch"], frame["val_f1_macro"], marker="o",
            label="val F1-macro", color="tab:blue",
        )
        if "val_f1_judi" in frame.columns:
            axes[1].plot(
                frame["epoch"], frame["val_f1_judi"], marker="s",
                label="val F1-judi", color="tab:orange",
            )
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("F1")
        axes[1].legend()
        axes[1].set_title("Validation F1")

        fig.suptitle(f"{scenario.upper()} run #{run_id}")
        return self._save(fig, f"{scenario}_run{run_id}_curve.png", dpi=110)

    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str,
        title: str = "Confusion Matrix",
    ) -> Path:
        """Gambar confusion matrix beranotasi angka.

        Args:
            y_true: Label sebenarnya.
            y_pred: Label prediksi.
            filename: Nama berkas PNG di folder gambar.
            title: Judul gambar.

        Returns:
            Path PNG yang ditulis.
        """
        matrix = self.evaluator.confusion(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4.5, 4))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)), labels=list(CLASS_NAMES))
        ax.set_yticks(range(len(CLASS_NAMES)), labels=list(CLASS_NAMES))
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title(title)

        threshold = matrix.max() / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, f"{matrix[i, j]:,}",
                    ha="center", va="center", fontweight="bold",
                    color="white" if matrix[i, j] > threshold else "black",
                )

        fig.colorbar(image, fraction=0.046, pad=0.04)
        return self._save(fig, filename)

    def pr_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        filename: str,
        title: str = "Precision-Recall (kelas judi)",
    ) -> Path:
        """Gambar PR curve kelas judi beserta average precision.

        Lebih informatif daripada confusion matrix tunggal pada data timpang
        4,5:1 karena memperlihatkan seluruh trade-off precision terhadap recall.

        Args:
            y_true: Label sebenarnya.
            y_score: Skor probabilitas kelas judi.
            filename: Nama berkas PNG di folder gambar.
            title: Judul gambar.

        Returns:
            Path PNG yang ditulis.
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        average_precision = self.evaluator.average_precision(y_true, y_score)
        baseline = float(y_true.mean()) if len(y_true) else 0.0

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(recall, precision, color="tab:purple")
        ax.axhline(
            baseline, color="gray", linestyle="--", linewidth=1,
            label=f"baseline ({baseline:.3f})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{title}\nAP={average_precision:.4f}")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left")
        return self._save(fig, filename)

    def grid_pivot_and_heatmap(
        self,
        scenario: str,
        metric: str = "val_f1_macro",
    ) -> list[Path]:
        """Pivot dan heatmap `metric` sebagai permukaan, satu gambar per nilai facet.

        Membaca grid sebagai permukaan memperlihatkan apakah nilai optimal satu
        sumbu bergeser mengikuti sumbu lain, yang tidak terlihat dari tabel datar.

        Args:
            scenario: Kode skenario.
            metric: Kolom metrik yang dipetakan.

        Returns:
            Daftar path CSV dan PNG yang ditulis; kosong bila variasi nilai
            belum cukup membentuk grid.
        """
        frame = self.runs_frame(scenario)
        axes = GRID_AXES.get(scenario)
        if frame.empty or axes is None:
            return []

        row_field, col_field, facet_field = axes
        required = (row_field, col_field, facet_field, metric)
        if not all(column in frame.columns for column in required):
            logger.debug("Heatmap %s dilewati: kolom %s tidak lengkap", scenario, required)
            return []

        frame = frame.dropna(subset=list(required))
        if frame[row_field].nunique() < 2 or frame[col_field].nunique() < 2:
            logger.debug("Heatmap %s dilewati: variasi nilai belum cukup", scenario)
            return []

        written: list[Path] = []
        for facet_value, subset in frame.groupby(facet_field):
            pivot = subset.pivot_table(
                index=row_field, columns=col_field, values=metric, aggfunc="max"
            )
            csv_path = self.metrics_dir / (
                f"{scenario}_grid_pivot_{facet_field}{facet_value}.csv"
            )
            write_csv(csv_path, pivot.reset_index())
            written.append(csv_path)

            fig, ax = plt.subplots(
                figsize=(1.2 * len(pivot.columns) + 2, 1.0 * len(pivot.index) + 2)
            )
            image = ax.imshow(pivot.values, cmap="viridis")
            ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), labels=[str(r) for r in pivot.index])
            ax.set_xlabel(col_field)
            ax.set_ylabel(row_field)
            ax.set_title(f"{scenario.upper()} {metric} -- {facet_field}={facet_value}")

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    value = pivot.values[i, j]
                    if not np.isnan(value):
                        ax.text(
                            j, i, f"{value:.3f}",
                            ha="center", va="center", color="white", fontsize=8,
                        )

            fig.colorbar(image, fraction=0.046, pad=0.04)
            written.append(
                self._save(
                    fig, f"{scenario}_grid_heatmap_{facet_field}{facet_value}.png"
                )
            )

        return written

    def tradeoff_scatter(
        self,
        scenario: str,
        x: str = "train_time_s",
        y: str = "val_f1_macro",
        size_by: str = "trainable_params",
        color_by: str = "peak_mem_mb",
    ) -> Path | None:
        """Scatter performa terhadap biaya DI DALAM satu skenario.

        Ukuran titik dan warnanya menambahkan dua dimensi efisiensi lagi tanpa
        gambar terpisah; keduanya diabaikan bila kolomnya tidak tersedia,
        misalnya RM-c yang tidak punya `peak_mem_mb`.

        Args:
            scenario: Kode skenario.
            x: Kolom sumbu biaya.
            y: Kolom sumbu performa.
            size_by: Kolom penentu ukuran titik.
            color_by: Kolom penentu warna titik.

        Returns:
            Path PNG, atau `None` bila baris kurang dari dua.
        """
        frame = self.runs_frame(scenario)
        if frame.empty or x not in frame.columns or y not in frame.columns:
            return None
        frame = frame.dropna(subset=[x, y])
        if len(frame) < 2:
            return None

        sizes = None
        if size_by in frame.columns and frame[size_by].notna().any():
            values = frame[size_by].fillna(frame[size_by].median())
            span = values.max() - values.min()
            if span > 0:
                sizes = 30 + 200 * (values - values.min()) / span

        colors = None
        if (
            color_by in frame.columns
            and frame[color_by].notna().any()
            and frame[color_by].nunique() > 1
        ):
            colors = frame[color_by]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        scatter = ax.scatter(
            frame[x],
            frame[y],
            s=60 if sizes is None else sizes,
            c="tab:blue" if colors is None else colors,
            cmap=None if colors is None else "plasma",
            alpha=0.8,
            edgecolors="black",
        )
        if colors is not None:
            fig.colorbar(scatter, ax=ax, label=color_by)

        if "run_id" in frame.columns:
            for _, row in frame.iterrows():
                ax.annotate(
                    str(int(row["run_id"])), (row[x], row[y]), fontsize=7, alpha=0.7
                )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        title = f"{scenario.upper()} -- {y} vs {x} (ukuran = {size_by}"
        title += f", warna = {color_by})" if colors is not None else ")"
        ax.set_title(title)
        return self._save(fig, f"{scenario}_tradeoff.png")

    def top_configs_bar_chart(
        self,
        scenario: str,
        metric: str = "val_f1_macro",
        top_n: int = 10,
    ) -> Path | None:
        """Bar chart horizontal N konfigurasi terbaik.

        Args:
            scenario: Kode skenario.
            metric: Kolom yang diurutkan.
            top_n: Jumlah konfigurasi yang ditampilkan.

        Returns:
            Path PNG, atau `None` bila riwayat kosong.
        """
        frame = self.runs_frame(scenario)
        if frame.empty or metric not in frame.columns:
            return None
        frame = frame.dropna(subset=[metric])
        if frame.empty:
            return None

        label_fields = [
            field for field in GRID_AXES.get(scenario, ())[:2] if field in frame.columns
        ]
        top = frame.nlargest(min(top_n, len(frame)), metric).sort_values(metric)
        labels = [self._config_label(row, label_fields) for _, row in top.iterrows()]

        fig, ax = plt.subplots(figsize=(6, 0.4 * len(top) + 1.5))
        ax.barh(range(len(top)), top[metric], color="tab:green")
        ax.set_yticks(range(len(top)), labels=labels, fontsize=8)
        ax.set_xlabel(metric)
        ax.set_title(f"{scenario.upper()} -- top {len(top)} konfigurasi by {metric}")
        return self._save(fig, f"{scenario}_top_configs.png")

    def final_inference_bar_chart(self) -> Path | None:
        """Bar chart latency dan memori inferensi RM-a/b/c dari benchmark satu sesi.

        Returns:
            Path PNG, atau `None` bila `metrics/inference_benchmark.csv` belum ada.
        """
        frame = read_csv(self.metrics_dir / "inference_benchmark.csv")
        if frame.empty or "scenario" not in frame.columns:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        palette = ["tab:blue", "tab:orange", "tab:green"][: len(frame)]

        if "infer_latency_ms" in frame.columns:
            axes[0].bar(frame["scenario"], frame["infer_latency_ms"], color=palette)
            axes[0].set_ylabel("ms/sampel")
            axes[0].set_title("Latency inferensi")
        else:
            axes[0].axis("off")

        if "infer_peak_gpu_mem_mb" in frame.columns:
            axes[1].bar(frame["scenario"], frame["infer_peak_gpu_mem_mb"], color=palette)
            axes[1].set_ylabel("MB")
            axes[1].set_title("Peak GPU memory inferensi")
        else:
            axes[1].axis("off")

        fig.suptitle("Benchmark inferensi RM-a/b/c (satu sesi GPU)")
        return self._save(fig, "final_inference_benchmark.png")

    def write_summary(self) -> Path:
        """Tulis `tuning_summary.json`: jumlah run, rentang waktu, dan juara tiap skenario.

        Returns:
            Path berkas ringkasan.

        Raises:
            CorruptArtifactError: Kalau `best.json` ada tapi rusak.
        """
        best = read_json(self.out_dir / "best.json", default={}) or {}
        summary: dict[str, object] = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenarios": {},
        }

        for scenario in SCENARIOS:
            frame = self.runs_frame(scenario)
            if frame.empty:
                continue

            entry: dict[str, object] = {"n_runs": len(frame)}
            if "timestamp" in frame.columns:
                entry["first_run_at"] = str(frame["timestamp"].min())
                entry["last_run_at"] = str(frame["timestamp"].max())
            if "batch_id" in frame.columns:
                batches = (
                    frame["batch_id"].dropna().astype(str).replace("", np.nan).dropna()
                )
                entry["n_batches"] = int(batches.nunique())
            if scenario in best:
                entry["best_run_id"] = best[scenario].get("run_id")
                entry["best_val_f1_macro"] = best[scenario].get("val_f1_macro")

            summary["scenarios"][scenario] = entry

        return write_json(self.out_dir / "tuning_summary.json", summary)

    def refresh_scenario(self, scenario: str) -> list[Path]:
        """Buat ulang seluruh gambar agregat satu skenario.

        Args:
            scenario: Kode skenario.

        Returns:
            Daftar path artefak yang berhasil dibuat.
        """
        written = list(self.grid_pivot_and_heatmap(scenario))
        for path in (
            self.tradeoff_scatter(scenario),
            self.top_configs_bar_chart(scenario),
        ):
            if path is not None:
                written.append(path)
        return written

    def _save(self, figure: plt.Figure, filename: str, dpi: int = FIGURE_DPI) -> Path:
        path = self.figures_dir / filename
        figure.tight_layout()
        try:
            figure.savefig(path, bbox_inches="tight", dpi=dpi)
        finally:
            plt.close(figure)
        return path

    @staticmethod
    def _config_label(row: pd.Series, fields: list[str]) -> str:
        parts = []
        for field in fields:
            value = row[field]
            if pd.isna(value):
                continue
            parts.append(
                f"{field}={value:g}" if isinstance(value, (int, float)) else f"{field}={value}"
            )
        run_id = (
            int(row["run_id"]) if "run_id" in row.index and pd.notna(row["run_id"]) else "?"
        )
        return f"#{run_id} " + " ".join(parts)
