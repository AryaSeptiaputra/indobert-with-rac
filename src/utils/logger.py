"""Setup logger dengan format text (development) atau JSON (production)."""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger.json import JsonFormatter

from src.config import settings

_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_JSON_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logger(name: str) -> logging.Logger:
    """Kembalikan logger bernama `name` dengan handler stdout terpasang.

    Handler ditulis ke stdout (bukan stderr) agar tertangkap utuh saat proses
    dijalankan sebagai subprocess yang stdout-nya dialihkan ke berkas log.

    Args:
        name: Nama logger, biasanya `__name__` modul pemanggil.

    Returns:
        Logger yang siap dipakai; pemanggilan berulang tidak menduplikasi handler.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)

    if settings.log_format == "json":
        formatter: logging.Formatter = JsonFormatter(_JSON_FORMAT)
    else:
        formatter = logging.Formatter(_TEXT_FORMAT)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
