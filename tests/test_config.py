"""Test konfigurasi terpusat dan penyelesaian path."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    CLASS_NAMES,
    LABEL_COLUMN,
    PROJECT_ROOT,
    SCENARIOS,
    SPECIAL_TOKENS,
    SPLIT_NAMES,
    TEXT_COLUMN,
    Settings,
    settings,
)


class TestSettings:
    def test_path_relatif_diselesaikan_ke_root_repo(self) -> None:
        """Nilai yang sama harus berlaku dari notebook maupun dari root repo."""
        assert settings.data_dir.is_absolute()
        assert settings.data_dir == PROJECT_ROOT / "data"

    def test_path_absolut_dibiarkan(self, tmp_path) -> None:
        custom = Settings(data_dir=tmp_path)
        assert custom.data_dir == tmp_path

    def test_struktur_folder_data_sesuai_standar(self) -> None:
        assert settings.raw_dir == settings.data_dir / "raw"
        assert settings.interim_dir == settings.data_dir / "interim"
        assert settings.processed_dir == settings.data_dir / "processed"

    def test_split_path_untuk_setiap_split(self) -> None:
        for split in SPLIT_NAMES:
            assert settings.split_path(split).name == f"{split}.csv"

    def test_split_tak_dikenal_ditolak(self) -> None:
        with pytest.raises(ValueError, match="split tak dikenal"):
            settings.split_path("holdout")

    def test_resolve_out_dir_relatif(self) -> None:
        assert settings.resolve_out_dir("outputs/coba") == PROJECT_ROOT / "outputs" / "coba"

    def test_resolve_out_dir_none_memakai_default(self) -> None:
        assert settings.resolve_out_dir(None) == settings.default_out_dir

    def test_resolve_out_dir_absolut_dibiarkan(self, tmp_path) -> None:
        assert settings.resolve_out_dir(tmp_path) == tmp_path

    def test_nilai_positif_ditegakkan(self) -> None:
        for field in ("micro_batch", "feature_batch_size", "latency_runs"):
            with pytest.raises(Exception):
                Settings(**{field: 0})

    def test_log_format_terbatas_pada_dua_pilihan(self) -> None:
        with pytest.raises(Exception):
            Settings(log_format="xml")


class TestKonstanta:
    def test_max_length_sesuai_keputusan_eda(self) -> None:
        """P99 panjang token 54; 128 menyisakan margin dengan truncation ~0,04%."""
        assert settings.max_length == 128

    def test_model_dasar_indobert_p2(self) -> None:
        assert settings.base_model == "indobenchmark/indobert-base-p2"

    def test_klasifikasi_biner(self) -> None:
        assert settings.num_labels == 2
        assert len(CLASS_NAMES) == 2

    def test_kolom_masukan_model_adalah_text_clean(self) -> None:
        """Memakai textOriginal akan melewati seluruh normalisasi preprocessing."""
        assert TEXT_COLUMN == "text_clean"
        assert LABEL_COLUMN == "label"

    def test_special_token_lengkap(self) -> None:
        assert SPECIAL_TOKENS == ("[URL]", "[MENTION]", "[NUM]")

    def test_tiga_skenario(self) -> None:
        assert SCENARIOS == ("rma", "rmb", "rmc")

    def test_tiga_split(self) -> None:
        assert SPLIT_NAMES == ("train", "val", "test")
