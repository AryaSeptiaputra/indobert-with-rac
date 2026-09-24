"""Statistik preprocessing untuk Tabel 4.2 dan 4.3, di luar area gate checksum.

Ditulis ke `outputs/preprocessing/preprocessing_stats.json`, BUKAN ke
`data/processed/metadata.json`, supaya menambah statistik tidak pernah mengubah
berkas yang dijaga gate reproduktibilitas di notebook 02.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.config import LABEL_COLUMN, RAW_TEXT_COLUMN, TEXT_COLUMN, settings
from src.services.preprocessing import DatasetBuilder
from src.utils.io import write_json

PP = 100.0
CLASS_KEYS = {0: "non_judi", 1: "judi"}


def unk_row_rate(texts: pd.Series, tokenizer) -> float:
    """Persentase teks yang memuat minimal satu token [UNK].

    Seluruh teks ditokenisasi tanpa pemotongan dan tanpa special token pembuka,
    supaya obfuskasi di mana pun di dalam komentar ikut terhitung.

    Args:
        texts: Kolom teks.
        tokenizer: Tokenizer pelatihan (sudah memuat [URL], [MENTION], [NUM]),
            sehingga placeholder tidak terhitung sebagai [UNK].

    Returns:
        Persentase baris, 0 sampai 100.
    """
    if texts.empty:
        return 0.0
    encoded = tokenizer(texts.tolist(), add_special_tokens=False, truncation=False)["input_ids"]
    unk = tokenizer.unk_token_id
    return sum(unk in ids for ids in encoded) / len(encoded) * PP


def nfkc_unk_rates(splits: dict[str, pd.DataFrame], builder: DatasetBuilder, tokenizer) -> dict[str, object]:
    """Tingkat [UNK] per kelas sebelum dan sesudah NFKC pada dataset final.

    "Sebelum" adalah pembersihan yang sama tanpa langkah NFKC; "sesudah" adalah
    `text_clean` yang dipakai pelatihan.
    """
    final = pd.concat(splits.values(), ignore_index=True)
    before = final[RAW_TEXT_COLUMN].map(lambda text: builder.cleaner.clean(text, apply_nfkc=False))
    rates: dict[str, dict[str, float]] = {"sebelum_nfkc": {}, "sesudah_nfkc": {}}
    for label, key in CLASS_KEYS.items():
        mask = final[LABEL_COLUMN] == label
        rates["sebelum_nfkc"][key] = unk_row_rate(before[mask], tokenizer)
        rates["sesudah_nfkc"][key] = unk_row_rate(final.loc[mask, TEXT_COLUMN], tokenizer)
    return {
        "unit": "persen baris yang memuat minimal satu [UNK]",
        "n_rows": {key: int((final[LABEL_COLUMN] == label).sum()) for label, key in CLASS_KEYS.items()},
        "rates_pct": rates,
    }


def write_preprocessing_stats(
    splits: dict[str, pd.DataFrame],
    builder: DatasetBuilder,
    tokenizer,
    path: str | Path | None = None,
) -> dict[str, object]:
    """Tulis jumlah baris per tahap, baris terbuang per kelas, dan dampak NFKC.

    Args:
        splits: Keluaran `DatasetBuilder.build`.
        builder: Builder yang sama, sudah memuat `counts` dan
            `leakage_removed_by_class`.
        tokenizer: Tokenizer pelatihan dari `load_tokenizer`.
        path: Tujuan; `None` memakai `outputs/preprocessing/preprocessing_stats.json`.

    Returns:
        Isi berkas yang ditulis.
    """
    path = Path(path) if path else settings.output_dir / "preprocessing" / "preprocessing_stats.json"
    removed_by_class: dict[str, int] = {}
    for per_split in builder.leakage_removed_by_class.values():
        for label, count in per_split.items():
            key = CLASS_KEYS[int(label)]
            removed_by_class[key] = removed_by_class.get(key, 0) + count

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage_counts": dict(builder.counts),
        "label_conflict": dict(builder.label_conflict),
        "leakage_removed_by_split_and_class": builder.leakage_removed_by_class,
        "leakage_removed_by_class": removed_by_class,
        "nfkc_unk": nfkc_unk_rates(splits, builder, tokenizer),
    }
    write_json(path, payload)
    return payload


__all__ = ["nfkc_unk_rates", "unk_row_rate", "write_preprocessing_stats"]
