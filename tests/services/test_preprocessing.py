"""Test pembersihan teks dan pembangunan split.

Determinisme split diuji secara eksplisit: seluruh angka eksperimen mengasumsikan
bahwa seed yang sama menghasilkan pembagian baris yang sama persis.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.services.preprocessing import DEDUP_KEY_COLUMN, DatasetBuilder, TextCleaner


class TestTextCleaner:
    def test_url_diganti_placeholder(self) -> None:
        cleaner = TextCleaner()
        assert cleaner.clean("cek https://judi.example/abc ya") == "cek [URL] ya"

    def test_www_dan_telegram_ikut_terdeteksi(self) -> None:
        cleaner = TextCleaner()
        assert cleaner.clean("buka www.slot.example") == "buka [URL]"
        assert cleaner.clean("gabung t.me/kanal") == "gabung [URL]"

    def test_mention_diganti_placeholder(self) -> None:
        assert TextCleaner().clean("halo @teman_saya") == "halo [MENTION]"

    def test_angka_berdiri_sendiri_diganti(self) -> None:
        assert TextCleaner().clean("deposit 50000 cair") == "deposit [NUM] cair"

    def test_brand_alfanumerik_dipertahankan(self) -> None:
        """DORA77 adalah nama situs, bukan angka; menggantinya menghapus sinyal."""
        assert TextCleaner().clean("DORA77 gacor") == "DORA77 gacor"

    def test_nfkc_menormalkan_fullwidth(self) -> None:
        assert TextCleaner().normalize_nfkc("ＶＩＤＥＯ") == "VIDEO"

    def test_karakter_tak_terlihat_dibuang(self) -> None:
        """Zero-width space dipakai untuk memecah kata agar lolos filter kata kunci."""
        assert TextCleaner().normalize_nfkc("ju​di") == "judi"

    def test_whitespace_dirapikan(self) -> None:
        assert TextCleaner().clean("  banyak   spasi \n\t ") == "banyak spasi"

    def test_emoji_dan_tanda_baca_dipertahankan(self) -> None:
        assert TextCleaner().clean("mantap! 🔥") == "mantap! 🔥"

    def test_tanpa_lowercase(self) -> None:
        """Tokenizer p2 sudah uncased; lowercase manual akan menggandakan pekerjaan."""
        assert TextCleaner().clean("Halo Dunia") == "Halo Dunia"

    def test_clean_series(self) -> None:
        result = TextCleaner().clean_series(pd.Series(["@a", "https://b.example"]))
        assert result.tolist() == ["[MENTION]", "[URL]"]


class TestDatasetBuilder:
    def test_ratios_harus_berjumlah_satu(self) -> None:
        with pytest.raises(ValueError, match="berjumlah 1"):
            DatasetBuilder(ratios=(0.5, 0.3, 0.3))

    def test_dedup_nfkc_menggabungkan_varian_unicode(self, raw_frame: pd.DataFrame) -> None:
        builder = DatasetBuilder()
        deduped = builder.deduplicate(raw_frame)
        keys = deduped[DEDUP_KEY_COLUMN].tolist()
        assert len(keys) == len(set(keys))
        # "ＶＩＤＥＯ keren" dan "VIDEO keren" runtuh menjadi satu baris.
        assert sum(1 for key in keys if key == "VIDEO keren") == 1

    def test_konflik_label_diselesaikan_ke_satu(self) -> None:
        frame = pd.DataFrame(
            {"textOriginal": ["teks sama", "teks sama"], "label": [0, 1]}
        )
        builder = DatasetBuilder()
        deduped = builder.deduplicate(frame)
        assert deduped["label"].tolist() == [1]
        assert builder.label_conflict["groups"] == 1
        assert builder.label_conflict["rows"] == 2

    def test_kolom_hilang_ditolak(self) -> None:
        with pytest.raises(KeyError, match="textOriginal"):
            DatasetBuilder().deduplicate(pd.DataFrame({"lain": ["a"], "label": [0]}))

    def test_split_deterministik(self, raw_frame: pd.DataFrame) -> None:
        """Seed yang sama harus menghasilkan pembagian baris yang identik."""
        first = DatasetBuilder(seed=42).build(raw_frame)
        second = DatasetBuilder(seed=42).build(raw_frame)
        for name in ("train", "val", "test"):
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_seed_berbeda_menghasilkan_split_berbeda(self, raw_frame: pd.DataFrame) -> None:
        first = DatasetBuilder(seed=42).build(raw_frame)
        second = DatasetBuilder(seed=7).build(raw_frame)
        assert first["train"]["textOriginal"].tolist() != second["train"]["textOriginal"].tolist()

    def test_build_menghasilkan_kolom_text_clean(self, raw_frame: pd.DataFrame) -> None:
        splits = DatasetBuilder().build(raw_frame)
        for frame in splits.values():
            assert "text_clean" in frame.columns
            assert not frame["text_clean"].isna().any()

    def test_tidak_ada_kebocoran_text_clean_lintas_split(
        self, raw_frame: pd.DataFrame
    ) -> None:
        """Setelah placeholder diterapkan, dua teks berbeda bisa jadi identik.

        Baris seperti itu harus dibuang dari val/test, bukan dibiarkan menjadi
        sampel evaluasi yang teks efektifnya sudah dilihat saat latih.
        """
        splits = DatasetBuilder().build(raw_frame)
        train = set(splits["train"]["text_clean"])
        assert not train & set(splits["val"]["text_clean"])
        assert not train & set(splits["test"]["text_clean"])
        assert not set(splits["val"]["text_clean"]) & set(splits["test"]["text_clean"])

    def test_train_tidak_pernah_dikurangi_guard_kebocoran(
        self, raw_frame: pd.DataFrame
    ) -> None:
        builder = DatasetBuilder()
        deduped = builder.deduplicate(raw_frame)
        train_before, _, _ = builder.stratified_split(deduped)
        splits = DatasetBuilder().build(raw_frame)
        assert len(splits["train"]) == len(train_before)

    def test_counts_konsisten(self, raw_frame: pd.DataFrame) -> None:
        builder = DatasetBuilder()
        splits = builder.build(raw_frame)
        counts = builder.counts
        assert counts["raw"] == len(raw_frame)
        assert counts["final_total"] == sum(len(frame) for frame in splits.values())
        assert (
            counts["after_dedup"] - counts["leakage_removed"] == counts["final_total"]
        )

    def test_class_weight_kelas_minoritas_lebih_besar(self) -> None:
        builder = DatasetBuilder()
        weights = builder.compute_class_weights([0] * 80 + [1] * 20)
        assert weights[1] > weights[0]
        assert weights[0] == pytest.approx(100 / (2 * 80))
        assert weights[1] == pytest.approx(100 / (2 * 20))

    def test_metadata_memuat_field_yang_dikutip_dokumen(
        self, raw_frame: pd.DataFrame
    ) -> None:
        builder = DatasetBuilder()
        splits = builder.build(raw_frame)
        weights = builder.compute_class_weights(splits["train"]["label"])
        metadata = builder.metadata(splits, weights)

        assert set(metadata["counts"]) >= {
            "raw", "after_missing", "after_dedup", "leakage_removed", "final_total"
        }
        assert set(metadata["splits"]) == {"train", "val", "test"}
        assert metadata["config"]["dedup_key"] == "nfkc_exact"
        assert metadata["config"]["split_ratios"] == [0.70, 0.15, 0.15]
        assert metadata["config"]["lowercase_manual"] is False

    def test_write_menghasilkan_seluruh_artefak(self, raw_frame, tmp_path) -> None:
        builder = DatasetBuilder()
        splits = builder.build(raw_frame)
        written = builder.write(
            splits,
            output_dir=tmp_path / "processed",
            interim_path=tmp_path / "interim" / "data_clean.csv",
        )
        for key in ("train", "val", "test", "metadata", "combined"):
            assert written[key].exists()

        combined = pd.read_csv(written["combined"])
        assert set(combined["split"]) == {"train", "val", "test"}
        assert len(combined) == sum(len(frame) for frame in splits.values())
