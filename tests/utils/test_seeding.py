"""Test penguncian seed lintas pustaka acak."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.config import settings
from src.utils.seeding import set_seed


class TestSetSeed:
    def test_mengembalikan_seed_default_dari_konfigurasi(self) -> None:
        assert set_seed() == settings.random_seed

    def test_mengembalikan_seed_eksplisit(self) -> None:
        assert set_seed(7) == 7

    def test_python_random_deterministik(self) -> None:
        set_seed(42)
        first = [random.random() for _ in range(5)]
        set_seed(42)
        assert [random.random() for _ in range(5)] == first

    def test_numpy_deterministik(self) -> None:
        set_seed(42)
        first = np.random.rand(5)
        set_seed(42)
        np.testing.assert_array_equal(np.random.rand(5), first)

    def test_torch_deterministik(self) -> None:
        """Inisialisasi bobot head dan urutan shuffle keduanya bergantung ini."""
        set_seed(42)
        first = torch.randn(5)
        set_seed(42)
        torch.testing.assert_close(torch.randn(5), first)

    def test_seed_berbeda_menghasilkan_nilai_berbeda(self) -> None:
        set_seed(1)
        first = torch.randn(5)
        set_seed(2)
        assert not torch.allclose(torch.randn(5), first)

    def test_mengunci_ketiga_pustaka_sekaligus(self) -> None:
        set_seed(123)
        snapshot = (random.random(), float(np.random.rand()), float(torch.randn(1)))
        set_seed(123)
        assert (random.random(), float(np.random.rand()), float(torch.randn(1))) == snapshot
