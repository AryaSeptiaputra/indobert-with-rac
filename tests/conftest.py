"""Fixture bersama: data sintetis kecil yang berjalan cepat di CPU."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.services.features import FeatureSet

FEATURE_DIM = 32
CLASS_WEIGHTS = (0.611, 2.752)


@pytest.fixture
def rng() -> np.random.Generator:
    """Generator acak dengan seed tetap agar test deterministik."""
    return np.random.default_rng(20260902)


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """DataFrame mentah yang memuat setiap kasus preprocessing sekaligus.

    Dua belas baris pertama adalah kasus uji yang disusun tangan: duplikat NFKC,
    konflik label, URL, mention, angka berdiri sendiri, brand alfanumerik, dan
    karakter tak-terlihat. Sisanya baris pengisi agar jumlahnya cukup untuk
    stratified split 70/15/15 -- scikit-learn menolak stratifikasi bila ada
    kelas dengan kurang dari dua anggota di suatu tahap.
    """
    cases = [
        ("Main di situs ini https://judi.example untuk menang", 1),
        ("Main di situs ini https://judi.example/lain untuk menang", 1),
        ("Halo @teman apa kabar", 0),
        ("Deposit 50000 langsung cair", 1),
        ("DORA77 gacor parah", 1),
        ("Video bagus sekali", 0),
        ("Video bagus sekali", 0),
        ("ＶＩＤＥＯ keren", 0),
        ("VIDEO keren", 0),
        ("Terima kasih ilmunya", 0),
        ("Klik www.slot.example sekarang", 1),
        ("Nonton terus sampai habis", 0),
    ]
    # Rasio pengisi mendekati ketimpangan data asli (sekitar 4,5:1).
    filler = [(f"komentar biasa nomor {i}", 0) for i in range(56)]
    filler += [(f"promo situs LUCKY{i} deposit murah", 1) for i in range(12)]
    return pd.DataFrame(cases + filler, columns=["textOriginal", "label"])


@pytest.fixture
def split_frames(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Tiga split kecil dengan kolom yang sama seperti data asli."""
    frames = {}
    for name, size in (("train", 60), ("val", 20), ("test", 20)):
        labels = rng.integers(0, 2, size)
        frames[name] = pd.DataFrame(
            {
                "textOriginal": [f"komentar mentah {name} {i}" for i in range(size)],
                "text_clean": [f"komentar {name} {i}" for i in range(size)],
                "label": labels,
            }
        )
    return frames


@pytest.fixture
def feature_set(rng: np.random.Generator) -> FeatureSet:
    """Embedding sintetis dengan sinyal kelas yang bisa dipelajari head."""
    embeddings, labels = {}, {}
    for name, size in (("train", 200), ("val", 60), ("test", 60)):
        y = rng.integers(0, 2, size).astype(np.int64)
        x = rng.normal(size=(size, FEATURE_DIM)).astype(np.float32)
        x[y == 1] += 1.2
        embeddings[name], labels[name] = x, y
    return FeatureSet(
        embeddings=embeddings,
        labels=labels,
        hidden_dim=FEATURE_DIM,
        model_name="sintetis",
        extract_time_s=1.5,
    )


@pytest.fixture
def class_weights() -> torch.Tensor:
    """Bobot kelas seperti pada dataset asli (rasio sekitar 4,5:1)."""
    return torch.tensor(CLASS_WEIGHTS, dtype=torch.float)


@pytest.fixture
def cpu_device() -> torch.device:
    """Device CPU agar test tidak bergantung pada ketersediaan GPU."""
    return torch.device("cpu")
