"""Penguncian seed untuk reproduktibilitas eksperimen."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.config import settings


def set_seed(seed: int | None = None) -> int:
    """Kunci seed `random`, `numpy`, dan `torch` (termasuk seluruh device CUDA).

    Args:
        seed: Nilai seed; `None` memakai `settings.random_seed`.

    Returns:
        Seed yang benar-benar dipakai.
    """
    effective = settings.random_seed if seed is None else int(seed)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)
    return effective
