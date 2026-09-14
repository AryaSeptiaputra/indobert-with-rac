"""Pemuatan split siap latih beserta tokenizer, class weight, dan device.

Sebelumnya semua ini dibangun sebagai dict tanpa tipe lalu dioper ke empat belas
fungsi berbeda. Sekarang menjadi satu objek dengan atribut bernama, sehingga
salah ketik nama kunci ketahuan saat impor, bukan di tengah training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.config import LABEL_COLUMN, SPLIT_NAMES, TEXT_COLUMN, settings
from src.models.comment_dataset import load_tokenizer
from src.utils.io import CorruptArtifactError, read_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Ukuran subset saat mode smoke; cukup untuk membuktikan pipeline berjalan
# ujung ke ujung tanpa menunggu satu epoch penuh.
SMOKE_SIZES = {"train": 200, "val": 100, "test": 100}


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Tentukan device komputasi.

    Args:
        device: Device eksplisit; `None` memilih CUDA bila tersedia.

    Returns:
        `torch.device` yang akan dipakai.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ExperimentData:
    """Split, tokenizer, dan class weight untuk satu sesi eksperimen.

    Attributes:
        frames: Peta nama split ke DataFrame-nya.
        tokenizer: Tokenizer dengan special token terdaftar.
        class_weights: Tensor bobot kelas di device yang sama dengan model.
        device: Device komputasi.
        model_name: Nama encoder dasar yang dipakai.
        max_length: Panjang token maksimum.
        smoke: True bila split sedang disubset untuk uji cepat.
    """

    frames: dict[str, pd.DataFrame]
    tokenizer: PreTrainedTokenizerBase
    class_weights: torch.Tensor
    device: torch.device
    model_name: str
    max_length: int
    smoke: bool = False
    num_workers: int = field(default_factory=lambda: settings.num_workers)

    @property
    def train(self) -> pd.DataFrame:
        return self.frames["train"]

    @property
    def val(self) -> pd.DataFrame:
        return self.frames["val"]

    @property
    def test(self) -> pd.DataFrame:
        return self.frames["test"]

    @classmethod
    def load(
        cls,
        model_name: str | None = None,
        device: str | torch.device | None = None,
        smoke: bool = False,
        processed_dir: Path | None = None,
    ) -> ExperimentData:
        """Muat ketiga split, tokenizer, dan class weight.

        Args:
            model_name: Encoder dasar; `None` memakai `settings.base_model`.
            device: Device komputasi; `None` memilih otomatis.
            smoke: Bila True, tiap split disubset (seed tetap) untuk uji cepat.
            processed_dir: Folder split; `None` memakai `settings.processed_dir`.

        Returns:
            Instance `ExperimentData` siap pakai.

        Raises:
            FileNotFoundError: Kalau ada berkas split atau metadata yang hilang.
            CorruptArtifactError: Kalau `metadata.json` tidak bisa diurai.
            KeyError: Kalau metadata tidak memuat `class_weights`.
        """
        processed_dir = processed_dir or settings.processed_dir
        resolved_device = resolve_device(device)
        name = model_name or settings.base_model

        frames: dict[str, pd.DataFrame] = {}
        for split in SPLIT_NAMES:
            path = processed_dir / f"{split}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"split {split!r} tidak ditemukan di {path}; "
                    "jalankan notebook 02_preprocessing lebih dulu"
                )
            frames[split] = pd.read_csv(path)

        if smoke:
            frames = {
                split: frame.sample(
                    min(len(frame), SMOKE_SIZES[split]),
                    random_state=settings.random_seed,
                ).reset_index(drop=True)
                for split, frame in frames.items()
            }
            logger.info(
                "Mode smoke: subset %s", {s: len(f) for s, f in frames.items()}
            )

        metadata_path = processed_dir / "metadata.json"
        metadata = read_json(metadata_path)
        if metadata is None:
            raise FileNotFoundError(f"metadata dataset tidak ditemukan di {metadata_path}")
        try:
            weights = metadata["class_weights"]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"'class_weights' tidak ada di {metadata_path}") from exc

        class_weights = torch.tensor(
            [float(weights["0"]), float(weights["1"])],
            dtype=torch.float,
            device=resolved_device,
        )

        logger.info(
            "Data dimuat: train=%d val=%d test=%d | device=%s | encoder=%s",
            len(frames["train"]),
            len(frames["val"]),
            len(frames["test"]),
            resolved_device,
            name,
        )

        return cls(
            frames=frames,
            tokenizer=load_tokenizer(name),
            class_weights=class_weights,
            device=resolved_device,
            model_name=name,
            max_length=settings.max_length,
            smoke=smoke,
        )

    def texts(self, split: str) -> list[str]:
        """Kolom teks bersih satu split.

        Args:
            split: Nama split.

        Returns:
            Daftar teks siap tokenisasi.
        """
        return self.frames[split][TEXT_COLUMN].tolist()

    def labels(self, split: str) -> list[int]:
        """Kolom label satu split.

        Args:
            split: Nama split.

        Returns:
            Daftar label int.
        """
        return self.frames[split][LABEL_COLUMN].tolist()


__all__ = ["ExperimentData", "CorruptArtifactError", "resolve_device", "SMOKE_SIZES"]
