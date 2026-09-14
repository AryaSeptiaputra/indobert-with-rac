"""Konfigurasi terpusat, dibaca dari environment / file `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Seluruh path, nama model, dan nilai default eksperimen.

    Path relatif diselesaikan terhadap root repositori, sehingga nilai yang sama
    berlaku baik saat dipanggil dari notebook, dari `app.py`, maupun dari
    subprocess worker.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("models")
    default_out_dir: Path = Path("outputs/tuning")

    base_model: str = "indobenchmark/indobert-base-p2"
    max_length: int = 128
    num_labels: int = 2

    random_seed: int = 42
    micro_batch: int = Field(default=8, gt=0)
    feature_batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=0, ge=0)

    latency_warmup_runs: int = Field(default=10, gt=0)
    latency_runs: int = Field(default=100, gt=0)

    tie_threshold_pp: float = Field(default=0.15, ge=0.0)

    log_format: Literal["text", "json"] = "text"

    @field_validator("data_dir", "output_dir", "model_dir", "default_out_dir")
    @classmethod
    def _resolve_against_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def raw_csv(self) -> Path:
        return self.raw_dir / "data_labeling.csv"

    @property
    def clean_csv(self) -> Path:
        return self.interim_dir / "data_clean.csv"

    @property
    def metadata_path(self) -> Path:
        return self.processed_dir / "metadata.json"

    def split_path(self, split: str) -> Path:
        """Path CSV untuk satu split.

        Args:
            split: Salah satu dari "train", "val", "test".

        Returns:
            Path ke file CSV split tersebut.

        Raises:
            ValueError: Kalau `split` bukan salah satu nama yang dikenal.
        """
        if split not in SPLIT_NAMES:
            raise ValueError(f"split tak dikenal: {split!r} (harus salah satu {SPLIT_NAMES})")
        return self.processed_dir / f"{split}.csv"

    def resolve_out_dir(self, out_dir: str | Path | None = None) -> Path:
        """Selesaikan `out_dir` job terhadap root repositori.

        Args:
            out_dir: Path folder keluaran; relatif diartikan terhadap root repo.
                `None` berarti memakai `default_out_dir`.

        Returns:
            Path absolut folder keluaran.
        """
        if out_dir is None:
            return self.default_out_dir
        path = Path(out_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
SCENARIOS: tuple[str, str, str] = ("rma", "rmb", "rmc")
TEXT_COLUMN = "text_clean"
RAW_TEXT_COLUMN = "textOriginal"
LABEL_COLUMN = "label"
CLASS_NAMES: tuple[str, str] = ("non-judi (0)", "judi (1)")

# Placeholder hasil preprocessing. Didaftarkan sebagai additional_special_tokens
# agar tiap placeholder menjadi satu token utuh, bukan terpecah "[", "url", "]".
URL_PLACEHOLDER = "[URL]"
MENTION_PLACEHOLDER = "[MENTION]"
NUM_PLACEHOLDER = "[NUM]"
SPECIAL_TOKENS: tuple[str, str, str] = (
    URL_PLACEHOLDER,
    MENTION_PLACEHOLDER,
    NUM_PLACEHOLDER,
)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instance `Settings` tunggal untuk seluruh proses."""
    return Settings()


settings = get_settings()
