"""Tokenizer dan PyTorch Dataset untuk komentar YouTube berlabel judi/non-judi.

Keputusan tokenisasi dikunci sejak fase EDA: `indobert-base-p2` (uncased),
`max_length` 128 (P99 panjang token = 54, truncation ~0,04%), dan placeholder
[URL]/[MENTION]/[NUM] terdaftar sebagai special token.

Karena special token menambah baris embedding, setiap model WAJIB memanggil
`resize_token_embeddings(len(tokenizer))` setelah memuat bobot pra-latih. Hal itu
sudah ditangani oleh factory di `src.models.heads`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.config import LABEL_COLUMN, SPECIAL_TOKENS, TEXT_COLUMN, settings
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    import pandas as pd

logger = setup_logger(__name__)

# Seluruh checkpoint indobenchmark (base maupun lite) punya vocab WordPiece
# >20 ribu entri. Ambang ini mendeteksi kegagalan senyap AutoTokenizer.
_MIN_SANE_VOCAB = 1000


@lru_cache(maxsize=4)
def load_tokenizer(model_name: str | None = None) -> PreTrainedTokenizerBase:
    """Muat tokenizer dan daftarkan placeholder sebagai special token.

    Repo Hub `indobenchmark/indobert-lite-*` berarsitektur ALBERT sehingga
    `AutoTokenizer` menebak kelas `AlbertTokenizer` yang berbasis SentencePiece,
    padahal repo itu hanya menyediakan `vocab.txt` WordPiece. Alih-alih gagal,
    tokenizer jatuh ke vocab minimal 5 token dan seluruh kata menjadi `[UNK]`.
    Kegagalan senyap itu dideteksi lewat ukuran vocab lalu ditambal dengan
    fallback eksplisit ke `BertTokenizer`.

    Args:
        model_name: Nama repo Hub; `None` memakai `settings.base_model`.

    Returns:
        Tokenizer dengan [URL]/[MENTION]/[NUM] terdaftar. Ukuran vocab bertambah
        tiga, sehingga model pemakainya wajib `resize_token_embeddings`.

    Raises:
        OSError: Kalau repo Hub tidak bisa diunduh atau tidak ditemukan.
        ValueError: Kalau vocab tetap tidak wajar walau sudah fallback.
    """
    name = model_name or settings.base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
    except OSError:
        logger.error("Gagal memuat tokenizer '%s' dari Hub", name, exc_info=True)
        raise

    if len(tokenizer) < _MIN_SANE_VOCAB:
        logger.warning(
            "AutoTokenizer untuk '%s' menghasilkan vocab mencurigakan kecil (%d); "
            "fallback ke BertTokenizer (WordPiece)",
            name,
            len(tokenizer),
        )
        tokenizer = BertTokenizer.from_pretrained(name)
        if len(tokenizer) < _MIN_SANE_VOCAB:
            raise ValueError(
                f"Tokenizer '{name}' tetap punya vocab kecil ({len(tokenizer)}) "
                "walau sudah fallback ke BertTokenizer; periksa repo Hub-nya."
            )

    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    return tokenizer


class GamblingCommentDataset(Dataset):
    """Dataset klasifikasi biner komentar (0 = non-judi, 1 = promosi judi).

    Tokenisasi dilakukan saat item diambil, bukan di muka, agar hemat memori dan
    selalu konsisten dengan tokenizer yang dipakai model.

    Args:
        texts: Teks komentar; gunakan kolom `text_clean` hasil preprocessing.
        tokenizer: Hasil `load_tokenizer()`.
        labels: Label int 0/1. `None` untuk inferensi tanpa label.
        max_length: Panjang token maksimum; `None` memakai `settings.max_length`.

    Raises:
        ValueError: Kalau jumlah label tidak sama dengan jumlah teks.
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: PreTrainedTokenizerBase,
        labels: Iterable[int] | None = None,
        max_length: int | None = None,
    ) -> None:
        self.texts: list[str] = [str(text) for text in texts]
        self.labels: list[int] | None = None if labels is None else [int(y) for y in labels]
        self.tokenizer = tokenizer
        self.max_length = settings.max_length if max_length is None else max_length

        if self.labels is not None and len(self.labels) != len(self.texts):
            raise ValueError(
                f"jumlah label ({len(self.labels)}) tidak sama dengan "
                f"jumlah teks ({len(self.texts)})"
            )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str | None = None,
        label_column: str | None = None,
        max_length: int | None = None,
    ) -> GamblingCommentDataset:
        """Bangun dataset dari DataFrame split.

        Args:
            frame: DataFrame berisi kolom teks dan (opsional) kolom label.
            tokenizer: Hasil `load_tokenizer()`.
            text_column: Nama kolom teks; `None` memakai `TEXT_COLUMN`.
            label_column: Nama kolom label; `None` memakai `LABEL_COLUMN`.
                Kolom yang tidak ada di `frame` diperlakukan sebagai tanpa label.
            max_length: Panjang token maksimum.

        Returns:
            Instance dataset siap dibungkus `DataLoader`.

        Raises:
            KeyError: Kalau kolom teks tidak ada di `frame`.
        """
        text_col = text_column or TEXT_COLUMN
        label_col = label_column or LABEL_COLUMN
        if text_col not in frame.columns:
            raise KeyError(f"kolom teks {text_col!r} tidak ada di DataFrame")
        labels: Sequence[int] | None = (
            frame[label_col].tolist() if label_col in frame.columns else None
        )
        return cls(
            texts=frame[text_col].tolist(),
            tokenizer=tokenizer,
            labels=labels,
            max_length=max_length,
        )
