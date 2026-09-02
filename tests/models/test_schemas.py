"""Test validasi hyperparameter."""

from __future__ import annotations

import pydantic
import pytest

from src.config import settings
from src.models.schemas import RMAConfig, RMBConfig, RMCConfig, RunRequest, parse_config


class TestRMAConfig:
    def test_default_adalah_baseline_kanonik(self) -> None:
        """Baseline IndoNLU/Wilie (2020), titik awal sekaligus run referensi grid."""
        config = RMAConfig()
        assert config.lr == pytest.approx(2e-5)
        assert config.epochs == 5
        assert config.batch == 16
        assert config.warmup_ratio == pytest.approx(0.1)
        assert config.weight_decay == pytest.approx(0.01)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("lr", 0.0),
            ("lr", -1e-5),
            ("epochs", 0),
            ("batch", 0),
            ("warmup_ratio", -0.1),
            ("warmup_ratio", 1.5),
            ("weight_decay", -0.01),
            ("micro_batch", 0),
        ],
    )
    def test_nilai_di_luar_batas_ditolak(self, field: str, value: float) -> None:
        with pytest.raises(pydantic.ValidationError):
            RMAConfig(**{field: value})

    def test_field_tak_dikenal_ditolak(self) -> None:
        """Salah ketik harus ketahuan di sini, bukan sebagai KeyError saat training."""
        with pytest.raises(pydantic.ValidationError):
            RMAConfig(learning_rate=2e-5)

    def test_akumulasi_gradien_mencapai_batch_efektif(self) -> None:
        config = RMAConfig(batch=32, micro_batch=8)
        assert config.effective_micro_batch == 8
        assert config.grad_accum == 4
        assert config.effective_micro_batch * config.grad_accum == 32

    def test_micro_batch_lebih_besar_dari_batch_dibatasi(self) -> None:
        config = RMAConfig(batch=8, micro_batch=32)
        assert config.effective_micro_batch == 8
        assert config.grad_accum == 1

    def test_micro_batch_default_dari_konfigurasi(self) -> None:
        assert RMAConfig().micro_batch == settings.micro_batch


class TestRMBConfig:
    def test_default_head_linear(self) -> None:
        assert RMBConfig().head_arch == "linear"

    def test_arsitektur_head_tak_dikenal_ditolak(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="linear"):
            RMBConfig(head_arch="cnn")

    def test_dropout_satu_ditolak(self) -> None:
        """Dropout 1,0 mematikan seluruh unit dan membuat head tidak bisa belajar."""
        with pytest.raises(pydantic.ValidationError):
            RMBConfig(dropout=1.0)

    def test_hidden_dim_di_atas_dimensi_masukan_diperbolehkan(self) -> None:
        assert RMBConfig(head_arch="mlp", hidden_dim=1024).hidden_dim == 1024


class TestRMCConfig:
    def test_default_alpha_dan_k(self) -> None:
        config = RMCConfig()
        assert config.alpha == pytest.approx(0.3)
        assert config.k == 5
        assert config.weighting == "similarity"

    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_alpha_di_luar_nol_satu_ditolak(self, alpha: float) -> None:
        with pytest.raises(pydantic.ValidationError):
            RMCConfig(alpha=alpha)

    @pytest.mark.parametrize("alpha", [0.0, 1.0])
    def test_alpha_di_ujung_rentang_diterima(self, alpha: float) -> None:
        assert RMCConfig(alpha=alpha).alpha == pytest.approx(alpha)

    def test_weighting_tak_dikenal_ditolak(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RMCConfig(weighting="cosine")


class TestParseConfig:
    def test_dict_kosong_memakai_default(self) -> None:
        assert parse_config("rma", {}).model_dump() == RMAConfig().model_dump()

    def test_none_memakai_default(self) -> None:
        assert parse_config("rmb").model_dump() == RMBConfig().model_dump()

    def test_nilai_parsial_menimpa_default_saja(self) -> None:
        config = parse_config("rma", {"lr": 5e-5})
        assert config.lr == pytest.approx(5e-5)
        assert config.epochs == RMAConfig().epochs

    def test_skenario_tak_dikenal_ditolak(self) -> None:
        with pytest.raises(ValueError, match="skenario tak dikenal"):
            parse_config("rmd", {})

    @pytest.mark.parametrize(
        "scenario,expected", [("rma", RMAConfig), ("rmb", RMBConfig), ("rmc", RMCConfig)]
    )
    def test_memetakan_ke_kelas_yang_benar(self, scenario: str, expected: type) -> None:
        assert isinstance(parse_config(scenario, {}), expected)


class TestRunRequest:
    def test_default_tidak_menyentuh_test_set(self) -> None:
        """Split test hanya dibuka sekali di benchmark final."""
        assert RunRequest().eval_test is False

    def test_menyimpan_catatan_apa_adanya(self) -> None:
        request = RunRequest(config={"lr": 3e-5}, note="naikkan lr, run 12 masih underfit")
        assert request.note == "naikkan lr, run 12 masih underfit"
