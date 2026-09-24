"""Metrik klasifikasi dan pengukuran efisiensi komputasi.

Dua tanggung jawab dipisah menjadi dua kelas: `ClassificationEvaluator` hanya
menghitung angka performa, `EfficiencyProfiler` hanya mengukur biaya komputasi.
Pembuatan gambar sepenuhnya berada di `src.services.reporting`.

Data ini timpang (rasio sekitar 4,5:1), sehingga accuracy saja menyesatkan.
Metrik utama adalah F1-macro, dengan F1 kelas 1 (judi) sebagai pemecah seri.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BYTES_PER_MB = 1024**2

# Kolom metrik yang disalin ke baris riwayat run. Satu sumber, sebelumnya daftar
# ini diketik ulang identik di tiga tempat.
RUN_METRIC_KEYS: tuple[str, ...] = (
    "f1_macro",
    "accuracy",
    "f1_class1",
    "precision_class1",
    "recall_class1",
)


class ClassificationEvaluator:
    """Hitung metrik klasifikasi biner untuk satu pasang (y_true, y_pred).

    Args:
        labels: Label kelas yang diperhitungkan, urut naik.
    """

    def __init__(self, labels: Sequence[int] = (0, 1)) -> None:
        self.labels = list(labels)

    def metrics(
        self,
        y_true: Sequence[int] | np.ndarray,
        y_pred: Sequence[int] | np.ndarray,
    ) -> dict[str, float | int]:
        """Metrik lengkap: macro, weighted, dan per-kelas.

        Args:
            y_true: Label sebenarnya.
            y_pred: Label prediksi.

        Returns:
            Dict metrik; kunci `f1_macro` adalah metrik utama penelitian dan
            `f1_class1` adalah F1 kelas judi.

        Raises:
            ValueError: Kalau panjang `y_true` dan `y_pred` berbeda.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"panjang y_true ({y_true.shape[0]}) != y_pred ({y_pred.shape[0]})"
            )

        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        p_class, r_class, f_class, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, zero_division=0
        )

        result: dict[str, float | int] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f_macro),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_weighted": float(f_weighted),
            "precision_weighted": float(p_weighted),
            "recall_weighted": float(r_weighted),
        }
        for index, label in enumerate(self.labels):
            result[f"f1_class{label}"] = float(f_class[index])
            result[f"precision_class{label}"] = float(p_class[index])
            result[f"recall_class{label}"] = float(r_class[index])
            result[f"support_class{label}"] = int(support[index])
        return result

    def confusion(
        self,
        y_true: Sequence[int] | np.ndarray,
        y_pred: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Confusion matrix dengan urutan label yang tetap.

        Args:
            y_true: Label sebenarnya.
            y_pred: Label prediksi.

        Returns:
            Array (n_labels, n_labels).
        """
        return confusion_matrix(y_true, y_pred, labels=self.labels)

    def average_precision(
        self,
        y_true: Sequence[int] | np.ndarray,
        y_score: Sequence[float] | np.ndarray,
    ) -> float:
        """Average precision kelas positif, ringkasan satu angka dari PR curve.

        Args:
            y_true: Label sebenarnya.
            y_score: Skor probabilitas kelas 1.

        Returns:
            Nilai average precision.
        """
        return float(average_precision_score(np.asarray(y_true), np.asarray(y_score)))

    def run_metrics(
        self,
        metrics: dict[str, float | int],
        prefix: str = "val",
    ) -> dict[str, float | int]:
        """Saring metrik menjadi kolom ringkas untuk baris riwayat run.

        Args:
            metrics: Keluaran `metrics`.
            prefix: Awalan nama kolom, biasanya "val" atau "test".

        Returns:
            Dict berkunci `{prefix}_{nama}` untuk metrik di `RUN_METRIC_KEYS`.
        """
        return {
            f"{prefix}_{key}": metrics[key]
            for key in RUN_METRIC_KEYS
            if key in metrics
        }


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CHUNK = 1_000
CONFIDENCE_LEVEL_PCT = 95.0


def _binary_f1_macro(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """F1-macro dua kelas per baris matriks resample."""
    scores = []
    for label in (0, 1):
        true_pos = ((truth == label) & (predicted == label)).sum(axis=1)
        false_pos = ((truth != label) & (predicted == label)).sum(axis=1)
        false_neg = ((truth == label) & (predicted != label)).sum(axis=1)
        denominator = 2 * true_pos + false_pos + false_neg
        scores.append(np.where(denominator > 0, 2 * true_pos / np.maximum(denominator, 1), 0.0))
    return np.mean(scores, axis=0)


def paired_bootstrap_f1(
    y_true: np.ndarray,
    challenger_pred: np.ndarray,
    incumbent_pred: np.ndarray,
    n_boot: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Selisih F1-macro penantang terhadap petahana dengan bootstrap berpasangan.

    Baris yang sama di-resample untuk kedua prediksi, sehingga variasi karena
    sampel sulit saling meniadakan. Hanya untuk klasifikasi biner.

    Args:
        y_true: Label sebenarnya (N,).
        challenger_pred: Prediksi penantang (N,).
        incumbent_pred: Prediksi petahana (N,).
        n_boot: Jumlah iterasi resample.
        seed: Seed generator acak.

    Returns:
        Dict `observed_delta_pp` (selisih pada data asli), `mean_delta_pp`
        (rata-rata selisih resample), `ci_low_pp` dan `ci_high_pp` (persentil
        95%), `n_boot`, `seed`, dan `n_samples`; semua selisih dalam poin
        persentase.

    Raises:
        ValueError: Kalau panjang ketiga array tidak sama.
    """
    y_true = np.asarray(y_true)
    challenger_pred = np.asarray(challenger_pred)
    incumbent_pred = np.asarray(incumbent_pred)
    if not len(y_true) == len(challenger_pred) == len(incumbent_pred):
        raise ValueError("y_true dan kedua prediksi harus sepanjang sama")

    everything = np.arange(len(y_true))[None, :]
    observed = (
        _binary_f1_macro(y_true[everything], challenger_pred[everything])[0]
        - _binary_f1_macro(y_true[everything], incumbent_pred[everything])[0]
    )

    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, n_boot, BOOTSTRAP_CHUNK):
        size = min(BOOTSTRAP_CHUNK, n_boot - start)
        rows = rng.integers(0, len(y_true), size=(size, len(y_true)))
        deltas.append(
            _binary_f1_macro(y_true[rows], challenger_pred[rows])
            - _binary_f1_macro(y_true[rows], incumbent_pred[rows])
        )
    deltas_pp = np.concatenate(deltas) * 100.0

    tail = (100.0 - CONFIDENCE_LEVEL_PCT) / 2.0
    low, high = np.percentile(deltas_pp, [tail, 100.0 - tail])
    return {
        "observed_delta_pp": float(observed * 100.0),
        "mean_delta_pp": float(deltas_pp.mean()),
        "ci_low_pp": float(low),
        "ci_high_pp": float(high),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_samples": int(len(y_true)),
    }


class EfficiencyProfiler:
    """Ukur biaya komputasi: jumlah parameter, memori GPU puncak, dan latency.

    Angka efisiensi hanya sebanding bila diukur pada satu hardware dan satu
    sesi. Jumlah warmup dan pengulangan diambil dari konfigurasi supaya seluruh
    skenario terukur dengan protokol yang sama; sebelumnya RM-b diukur 5/50
    saat tuning tapi 10/100 saat benchmark final.

    Args:
        n_warmup: Jumlah pemanggilan pemanasan; `None` memakai konfigurasi.
        n_runs: Jumlah pemanggilan terukur; `None` memakai konfigurasi.
    """

    def __init__(self, n_warmup: int | None = None, n_runs: int | None = None) -> None:
        self.n_warmup = settings.latency_warmup_runs if n_warmup is None else n_warmup
        self.n_runs = settings.latency_runs if n_runs is None else n_runs

    def count_parameters(self, model: torch.nn.Module) -> dict[str, float | int]:
        """Hitung total dan trainable parameter sebuah modul.

        Args:
            model: Modul PyTorch apa pun.

        Returns:
            Dict berisi `total_params`, `trainable_params`, dan `trainable_pct`.
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_params": int(total),
            "trainable_params": int(trainable),
            "trainable_pct": float(trainable / total * 100) if total else 0.0,
        }

    def reset_peak_memory(self) -> None:
        """Nolkan pencatat memori GPU puncak sebelum bagian yang diukur."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_gpu_memory_mb(self) -> float:
        """Memori GPU puncak (MB) sejak reset terakhir; 0 bila berjalan di CPU."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / BYTES_PER_MB
        return 0.0

    @torch.no_grad()
    def measure_stages(
        self,
        stages: Sequence[tuple[str, Callable[[object], object]]],
    ) -> dict[str, float]:
        """Latency rata-rata tiap tahap dari satu jalur inferensi berurutan.

        Tiap iterasi menjalankan seluruh tahap berurutan; keluaran satu tahap
        menjadi masukan tahap berikutnya (tahap pertama menerima None). CUDA
        disinkronkan di batas tiap tahap agar waktunya teratribusi benar, sehingga
        jumlah seluruh tahap adalah latency jalur itu.

        Args:
            stages: Pasangan (nama, fungsi satu argumen) sesuai urutan jalur.

        Returns:
            Latency rata-rata per tahap dalam milidetik, urutan dipertahankan.

        Raises:
            RuntimeError: Kalau salah satu tahap gagal saat pemanasan.
        """
        use_cuda = torch.cuda.is_available()

        def run_once(elapsed: dict[str, float] | None) -> None:
            value: object = None
            for name, stage in stages:
                if use_cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                value = stage(value)
                if use_cuda:
                    torch.cuda.synchronize()
                if elapsed is not None:
                    elapsed[name] += time.perf_counter() - started

        try:
            for _ in range(self.n_warmup):
                run_once(None)
        except RuntimeError:
            logger.error("Pemanasan pengukuran latency bertahap gagal", exc_info=True)
            raise

        elapsed = {name: 0.0 for name, _ in stages}
        for _ in range(self.n_runs):
            run_once(elapsed)
        return {name: total / self.n_runs * 1000.0 for name, total in elapsed.items()}

    def measure_latency(self, predict: Callable[[], object]) -> float:
        """Rata-rata waktu satu pemanggilan `predict` dalam milidetik.

        `predict` harus melakukan tepat satu forward pass untuk satu sampel dan
        tidak menerima argumen; sinkronisasi CUDA diperhitungkan bila di GPU.

        Args:
            predict: Callable tanpa argumen yang menjalankan satu inferensi.

        Returns:
            Latency rata-rata dalam milidetik per pemanggilan.

        Raises:
            RuntimeError: Kalau `predict` gagal saat pemanasan.
        """
        try:
            for _ in range(self.n_warmup):
                predict()
        except RuntimeError:
            logger.error("Pemanasan pengukuran latency gagal", exc_info=True)
            raise

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(self.n_runs):
            predict()
        if use_cuda:
            torch.cuda.synchronize()

        return (time.perf_counter() - start) / self.n_runs * 1000.0
