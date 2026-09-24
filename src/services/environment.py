"""Spesifikasi lingkungan eksperimen dan gate satu-hardware (Aturan #1).

`hardware.json` memuat spesifikasi lingkungan di tingkat atas, daftar sesi
tuning (`tuning_sessions`), dan sesi benchmark final (`final_session`). Waktu
boot mesin membedakan sesi: hardware yang sama dengan waktu boot berbeda berarti
sesi berbeda, dan itu dicatat apa adanya supaya Bab 4 bisa menyatakannya jujur.
"""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

import psutil
import torch

from src.config import settings
from src.utils.io import read_json, write_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BYTES_PER_MB = 1024**2
BYTES_PER_GB = 1024**3

# Berbeda di salah satu kunci ini berarti hardware atau pustaka berbeda: angka
# efisiensi tidak boleh digabung, dan benchmark final berhenti.
STRICT_KEYS = ("gpu", "vram_total_mb", "driver", "cuda_version", "torch", "transformers", "faiss")


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


__all__ = [
    "EnvironmentMismatchError",
    "STRICT_KEYS",
    "collect_environment",
    "record_tuning_session",
    "verify_final_session",
]
