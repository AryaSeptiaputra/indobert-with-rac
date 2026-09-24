"""Test statistik preprocessing untuk Tabel 4.2 dan 4.3."""

from __future__ import annotations

import json

import pandas as pd

from src.services.preprocessing import DatasetBuilder, TextCleaner
from src.services.preprocessing_stats import unk_row_rate, write_preprocessing_stats


class TokenizerPalsu:
    """Satu token per kata; kata ber-karakter non-ASCII menjadi [UNK]."""

    unk_token_id = 100

    def __call__(self, texts, add_special_tokens=False, truncation=False):
        return {"input_ids": [
            [self.unk_token_id if any(ord(ch) > 127 for ch in word) else 1 for word in text.split()]
            for text in texts
        ]}


def test_tanpa_nfkc_fullwidth_bertahan() -> None:
    cleaner = TextCleaner()
    assert cleaner.clean("ＤＯＲＡ７７ gacor", apply_nfkc=False) == "ＤＯＲＡ７７ gacor"
    assert cleaner.clean("ＤＯＲＡ７７ gacor") == "DORA77 gacor"


def test_tingkat_unk_persen_baris() -> None:
    texts = pd.Series(["aman saja", "𝐆𝐀𝐂𝐎𝐑 hari ini", "ok", "ＪＰ"])
    assert unk_row_rate(texts, TokenizerPalsu()) == 50.0


def test_statistik_ditulis_terpisah_dari_metadata(raw_frame, tmp_path) -> None:
    builder = DatasetBuilder(seed=42)
    splits = builder.build(raw_frame)

    path = tmp_path / "preprocessing_stats.json"
    payload = write_preprocessing_stats(splits, builder, TokenizerPalsu(), path=path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["stage_counts"] == builder.counts
    assert sum(saved["leakage_removed_by_class"].values()) == builder.counts["leakage_removed"]
    rates = payload["nfkc_unk"]["rates_pct"]
    assert set(rates) == {"sebelum_nfkc", "sesudah_nfkc"}
    assert set(rates["sesudah_nfkc"]) == {"judi", "non_judi"}
    assert rates["sesudah_nfkc"]["judi"] <= rates["sebelum_nfkc"]["judi"]
