"""Test FeatureSet dan mekanisme cache FeatureExtractor."""

from __future__ import annotations

import numpy as np
import pytest

from src.services.features import FeatureExtractor, FeatureSet, model_slug
from src.utils.io import read_json


class TestModelSlug:
    def test_garis_miring_diganti(self) -> None:
        assert model_slug("indobenchmark/indobert-base-p2") == "indobenchmark__indobert-base-p2"

    def test_nama_tanpa_garis_miring_tak_berubah(self) -> None:
        assert model_slug("sintetis") == "sintetis"


class TestFeatureSet:
    def test_akses_split_mengembalikan_pasangan(self, feature_set: FeatureSet) -> None:
        embeddings, labels = feature_set["train"]
        assert embeddings.shape[0] == labels.shape[0]
        assert embeddings.shape[1] == feature_set.hidden_dim

    def test_split_tak_dikenal_ditolak(self, feature_set: FeatureSet) -> None:
        with pytest.raises(KeyError):
            feature_set["holdout"]

    def test_ketiga_split_tersedia(self, feature_set: FeatureSet) -> None:
        assert set(feature_set.embeddings) == {"train", "val", "test"}


class TestCache:
    def _extractor(self) -> FeatureExtractor:
        import torch

        return FeatureExtractor(
            tokenizer=None, device=torch.device("cpu"), model_name="sintetis"
        )

    def test_cache_dipisah_per_encoder(self, tmp_path) -> None:
        """Dua encoder berbeda tidak boleh saling menimpa embedding."""
        import torch

        base = FeatureExtractor(None, torch.device("cpu"), model_name="encoder/a")
        lite = FeatureExtractor(None, torch.device("cpu"), model_name="encoder/b")
        assert base.cache_dir(tmp_path) != lite.cache_dir(tmp_path)

    def test_cache_kosong_mengembalikan_none(self, tmp_path) -> None:
        assert self._extractor()._load_cache(tmp_path / "kosong") is None

    def test_simpan_lalu_muat_mengembalikan_isi_sama(
        self, tmp_path, feature_set: FeatureSet
    ) -> None:
        extractor = self._extractor()
        cache = extractor.cache_dir(tmp_path)
        extractor._save_cache(cache, feature_set)

        loaded = extractor._load_cache(cache)
        assert loaded is not None
        assert loaded.from_cache is True
        assert loaded.hidden_dim == feature_set.hidden_dim
        for split in ("train", "val", "test"):
            np.testing.assert_array_equal(
                loaded.embeddings[split], feature_set.embeddings[split]
            )
            np.testing.assert_array_equal(loaded.labels[split], feature_set.labels[split])

    def test_metadata_cache_mencatat_encoder_asal(
        self, tmp_path, feature_set: FeatureSet
    ) -> None:
        extractor = self._extractor()
        cache = extractor.cache_dir(tmp_path)
        extractor._save_cache(cache, feature_set)

        meta = read_json(cache / "extract_meta.json")
        assert meta["model_name"] == "sintetis"
        assert meta["hidden_dim"] == feature_set.hidden_dim

    def test_cache_encoder_lain_ditolak(self, tmp_path, feature_set: FeatureSet) -> None:
        """Head yang dilatih di atas embedding encoder lain menghasilkan angka palsu."""
        import torch

        writer = FeatureExtractor(None, torch.device("cpu"), model_name="encoder/a")
        cache_a = writer.cache_dir(tmp_path)
        writer._save_cache(cache_a, feature_set)

        reader = FeatureExtractor(None, torch.device("cpu"), model_name="encoder/b")
        assert reader._load_cache(cache_a) is None

    def test_cache_tidak_lengkap_mengembalikan_none(
        self, tmp_path, feature_set: FeatureSet
    ) -> None:
        extractor = self._extractor()
        cache = extractor.cache_dir(tmp_path)
        extractor._save_cache(cache, feature_set)
        (cache / "test_emb.npy").unlink()
        assert extractor._load_cache(cache) is None

    def test_waktu_ekstraksi_ikut_tersimpan(
        self, tmp_path, feature_set: FeatureSet
    ) -> None:
        """Biaya ekstraksi adalah bagian jujur dari waktu latih RM-b."""
        extractor = self._extractor()
        cache = extractor.cache_dir(tmp_path)
        extractor._save_cache(cache, feature_set)
        assert extractor._load_cache(cache).extract_time_s == pytest.approx(1.5)
