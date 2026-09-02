"""Skema Pydantic untuk konfigurasi hyperparameter tiap skenario.

Sebelumnya nilai hyperparameter dioper sebagai dict telanjang tanpa validasi apa
pun, sehingga salah ketik baru ketahuan sebagai `KeyError` di tengah training.
Semua batas nilai sekarang ditegakkan di sini, sebelum satu epoch pun berjalan.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import settings

Scenario = Literal["rma", "rmb", "rmc"]
HeadArch = Literal["linear", "mlp"]
Weighting = Literal["similarity", "uniform"]


class RMAConfig(BaseModel):
    """Hyperparameter RM-a (full fine-tuning).

    Nilai default adalah baseline kanonik IndoNLU/Wilie (2020), dipakai sebagai
    titik awal sekaligus run referensi seluruh grid.
    """

    model_config = ConfigDict(extra="forbid")

    lr: float = Field(default=2e-5, gt=0.0)
    epochs: int = Field(default=5, ge=1)
    batch: int = Field(default=16, ge=1)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    micro_batch: int = Field(default_factory=lambda: settings.micro_batch, ge=1)
    seed: int = Field(default_factory=lambda: settings.random_seed)

    @property
    def effective_micro_batch(self) -> int:
        """Ukuran batch yang benar-benar masuk GPU per langkah maju/mundur."""
        return min(self.batch, self.micro_batch)

    @property
    def grad_accum(self) -> int:
        """Langkah akumulasi gradien yang dibutuhkan untuk mencapai `batch` efektif.

        Akumulasi gradien ekuivalen secara matematis dengan batch besar untuk
        BERT (LayerNorm, bukan BatchNorm) selama loss dibagi jumlah akumulasi,
        sehingga ini murni kompromi memori dan bukan kompromi hasil.
        """
        return max(1, self.batch // self.effective_micro_batch)


class RMBConfig(BaseModel):
    """Hyperparameter RM-b (encoder beku + classification head)."""

    model_config = ConfigDict(extra="forbid")

    head_arch: HeadArch = "linear"
    hidden_dim: int = Field(default=256, ge=1)
    epochs: int = Field(default=5, ge=1)
    lr: float = Field(default=2e-4, gt=0.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    batch: int = Field(default=32, ge=1)
    seed: int = Field(default_factory=lambda: settings.random_seed)


class RMCConfig(BaseModel):
    """Hyperparameter RM-c (RAC): bobot fusi, jumlah tetangga, skema pembobotan.

    `alpha` membobot cabang retrieval terhadap cabang BERT; `alpha=0` identik
    dengan RM-b murni dan `alpha=1` membuang cabang BERT sepenuhnya.
    """

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    k: int = Field(default=5, ge=1)
    weighting: Weighting = "similarity"


class RunRequest(BaseModel):
    """Satu permintaan run: hyperparameter mentah plus alasan mencobanya.

    `note` sengaja wajib dipertimbangkan manusia dan disimpan apa adanya ke kolom
    `catatan`, sehingga setiap nilai hyperparameter punya justifikasi eksplisit
    yang bisa dikutip saat menulis Bab 4.
    """

    model_config = ConfigDict(extra="forbid")

    config: dict[str, object] = Field(default_factory=dict)
    note: str = ""
    eval_test: bool = False


CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "rma": RMAConfig,
    "rmb": RMBConfig,
    "rmc": RMCConfig,
}


def parse_config(scenario: str, raw: dict[str, object] | None = None) -> BaseModel:
    """Validasi dict hyperparameter mentah menjadi model skenario yang sesuai.

    Args:
        scenario: Kode skenario ("rma", "rmb", atau "rmc").
        raw: Dict hyperparameter dari notebook atau CSV grid; `None` berarti
            memakai seluruh nilai default.

    Returns:
        Instance `RMAConfig`, `RMBConfig`, atau `RMCConfig`.

    Raises:
        ValueError: Kalau `scenario` tidak dikenal.
        pydantic.ValidationError: Kalau ada nilai di luar batas yang sah.
    """
    if scenario not in CONFIG_MODELS:
        raise ValueError(
            f"skenario tak dikenal: {scenario!r} (harus salah satu {tuple(CONFIG_MODELS)})"
        )
    return CONFIG_MODELS[scenario].model_validate(raw or {})
