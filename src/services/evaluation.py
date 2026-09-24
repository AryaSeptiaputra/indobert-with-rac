"""Metrik klasifikasi dan pengukuran efisiensi komputasi.

Dua tanggung jawab dipisah menjadi dua kelas: `ClassificationEvaluator` hanya
menghitung angka performa, `EfficiencyProfiler` hanya mengukur biaya komputasi.
Pembuatan gambar sepenuhnya berada di `src.services.reporting`.

Data ini timpang (rasio sekitar 4,5:1), sehingga accuracy saja menyesatkan.
Metrik utama adalah F1-macro, dengan F1 kelas 1 (judi) sebagai pemecah seri.

Spesifikasi lingkungan eksperimen dan gate satu-hardware (Aturan #1) juga ada di
sini. `hardware.json` memuat spesifikasi lingkungan di tingkat atas, daftar sesi
tuning (`tuning_sessions`), dan sesi benchmark final (`final_session`). Waktu
boot mesin membedakan sesi: hardware yang sama dengan waktu boot berbeda berarti
sesi berbeda, dan itu dicatat apa adanya supaya Bab 4 bisa menyatakannya jujur.
"""

from __future__ import annotations

import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import settings
from src.utils.io import read_json, write_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BYTES_PER_MB = 1024**2
BYTES_PER_GB = 1024**3

# Berbeda di salah satu kunci ini berarti hardware atau pustaka berbeda: angka
# efisiensi tidak boleh digabung, dan benchmark final berhenti.
STRICT_KEYS = ("gpu", "vram_total_mb", "driver", "cuda_version", "torch", "transformers", "faiss")

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


class EnvironmentMismatchError(RuntimeError):
    """Lingkungan benchmark final berbeda dari lingkungan tuning."""


def _nvidia_driver() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("nvidia-smi tidak tersedia; versi driver tidak tercatat")
        return None
    return output.strip().splitlines()[0] if output.strip() else None


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def collect_environment() -> dict[str, object]:
    """Spesifikasi lingkungan saat ini: GPU, driver, CUDA, CPU, RAM, OS, dan versi pustaka."""
    import faiss
    import transformers

    cuda = torch.cuda.is_available()
    return {
        "gpu": torch.cuda.get_device_name(0) if cuda else "CPU",
        "vram_total_mb": (
            round(torch.cuda.get_device_properties(0).total_memory / BYTES_PER_MB) if cuda else 0
        ),
        "driver": _nvidia_driver() if cuda else None,
        "cuda_version": torch.version.cuda,
        "cpu": _cpu_name(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / BYTES_PER_GB, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "faiss": faiss.__version__,
        "seed": settings.random_seed,
    }


def current_session() -> dict[str, object]:
    """Cap waktu sesi saat ini beserta waktu boot mesin."""
    return {
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "boot_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(psutil.boot_time())),
    }


def mismatches(recorded: dict[str, object], current: dict[str, object]) -> dict[str, list[object]]:
    """Kunci ketat yang nilainya berbeda: `{kunci: [tercatat, sekarang]}`."""
    return {
        key: [recorded.get(key), current.get(key)]
        for key in STRICT_KEYS
        if recorded.get(key) != current.get(key)
    }


def record_tuning_session(path: str | Path) -> dict[str, object]:
    """Catat lingkungan tuning; sesi baru ditambahkan bila waktu boot berbeda.

    Raises:
        EnvironmentMismatchError: Kalau `hardware.json` sudah memuat lingkungan
            lain. Kampanye di folder ini tidak boleh berlanjut di hardware atau
            versi pustaka yang berbeda.
    """
    path = Path(path)
    recorded = read_json(path, default={}) or {}
    environment = collect_environment()

    if recorded.get("gpu") is not None:
        differences = mismatches(recorded, environment)
        if differences:
            raise EnvironmentMismatchError(
                f"lingkungan berbeda dari yang tercatat di {path}: {differences}"
            )

    sessions = list(recorded.get("tuning_sessions", []))
    session = current_session()
    if session["boot_time"] not in {entry.get("boot_time") for entry in sessions}:
        sessions.append(session)
        if len(sessions) > 1:
            logger.warning("Tuning berlanjut di sesi mesin baru (boot %s)", session["boot_time"])

    payload = {**recorded, **environment, "tuning_sessions": sessions}
    write_json(path, payload)
    return payload


def verify_final_session(path: str | Path) -> dict[str, object]:
    """Gate benchmark final: lingkungan wajib identik dengan saat tuning.

    Returns:
        Isi `hardware.json` setelah `final_session` ditambahkan.

    Raises:
        FileNotFoundError: Kalau lingkungan tuning belum pernah dicatat.
        EnvironmentMismatchError: Kalau salah satu kunci ketat berbeda. Tidak ada
            flag untuk melewatinya.
    """
    path = Path(path)
    recorded = read_json(path, default={}) or {}
    if recorded.get("gpu") is None:
        raise FileNotFoundError(f"{path} belum memuat lingkungan tuning; jalankan 03a lebih dulu")

    differences = mismatches(recorded, collect_environment())
    if differences:
        raise EnvironmentMismatchError(
            f"benchmark final dihentikan: lingkungan berbeda dari saat tuning {differences}"
        )

    session = current_session()
    tuning_boots = [entry.get("boot_time") for entry in recorded.get("tuning_sessions", [])]
    session["same_boot_as_tuning"] = session["boot_time"] in tuning_boots
    if not session["same_boot_as_tuning"]:
        logger.warning(
            "Hardware sama tetapi sesi berbeda: boot final %s, boot tuning %s",
            session["boot_time"], tuning_boots,
        )

    payload = {**recorded, "final_session": session}
    write_json(path, payload)
    return payload
