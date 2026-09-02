"""Baca/tulis berkas JSON dan CSV dengan penanganan galat eksplisit.

Penulisan selalu atomik (tulis ke berkas sementara di folder tujuan lalu
`os.replace`), sehingga proses yang mati di tengah penulisan tidak meninggalkan
berkas hasil yang terpotong.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CorruptArtifactError(RuntimeError):
    """Berkas artefak ada tetapi isinya tidak bisa diurai."""


def read_json(path: str | Path, default: Any = None) -> Any:
    """Baca berkas JSON.

    Berkas yang belum ada mengembalikan `default`. Berkas yang ADA tetapi rusak
    justru dilaporkan sebagai galat, bukan diam-diam diganti `default` — kalau
    tidak, pemanggil akan menimpa artefak yang masih bisa diselamatkan.

    Args:
        path: Lokasi berkas JSON.
        default: Nilai yang dikembalikan bila berkas belum ada.

    Returns:
        Objek hasil parse, atau `default` bila berkas tidak ada.

    Raises:
        CorruptArtifactError: Kalau berkas ada tapi bukan JSON yang sah, atau
            tidak bisa dibaca.
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise CorruptArtifactError(f"JSON rusak di {path}: {exc}") from exc
    except OSError as exc:
        raise CorruptArtifactError(f"Gagal membaca {path}: {exc}") from exc


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Tulis `payload` sebagai JSON secara atomik.

    Args:
        path: Lokasi berkas tujuan; folder induk dibuat bila belum ada.
        payload: Objek yang bisa diserialkan JSON.
        indent: Indentasi keluaran.

    Returns:
        Path berkas yang ditulis.

    Raises:
        OSError: Kalau penulisan atau penggantian berkas gagal.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    _atomic_write_text(path, text)
    return path


def read_csv(path: str | Path, default: pd.DataFrame | None = None) -> pd.DataFrame:
    """Baca CSV; berkas yang belum ada mengembalikan `default` (atau DataFrame kosong).

    Args:
        path: Lokasi berkas CSV.
        default: DataFrame pengganti bila berkas belum ada.

    Returns:
        DataFrame hasil pembacaan.

    Raises:
        CorruptArtifactError: Kalau berkas ada tapi tidak bisa diurai atau dibaca.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame() if default is None else default
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame() if default is None else default
    except pd.errors.ParserError as exc:
        raise CorruptArtifactError(f"CSV rusak di {path}: {exc}") from exc
    except OSError as exc:
        raise CorruptArtifactError(f"Gagal membaca {path}: {exc}") from exc


def write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    """Tulis DataFrame ke CSV secara atomik (tanpa kolom index).

    Args:
        path: Lokasi berkas tujuan; folder induk dibuat bila belum ada.
        frame: DataFrame yang ditulis.

    Returns:
        Path berkas yang ditulis.

    Raises:
        OSError: Kalau penulisan atau penggantian berkas gagal.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, frame.to_csv(index=False))
    return path


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        logger.error("Penulisan atomik ke %s gagal", path, exc_info=True)
        raise
