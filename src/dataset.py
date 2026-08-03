"""PyTorch Dataset & tokenizer untuk IndoBERT-with-RAC.

Mengunci keputusan tokenisasi fase EDA:
- Base tokenizer: `indobenchmark/indobert-base-p2` (do_lower_case=True).
- max_length = 128 (P99 panjang token = 54; truncation ~0,04%).
- Placeholder [URL]/[MENTION]/[NUM] didaftarkan sebagai additional_special_tokens
  agar tiap placeholder menjadi SATU token utuh (bukan terpecah '[','url',']').

PENTING: karena special token menambah ukuran vocab, setiap model WAJIB memanggil
    model.resize_token_embeddings(len(tokenizer))
setelah memuat bobot pra-latih, sebelum training/inference. Berlaku untuk
RM-a (03), RM-b (04), dan RM-c (05).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer

from preprocessing import SPECIAL_TOKENS

MODEL_NAME = "indobenchmark/indobert-base-p2"
MAX_LENGTH = 128

# Ambang vocab yang masuk akal -- semua checkpoint indobenchmark (base maupun lite)
# punya vocab WordPiece >20rb. Dipakai untuk mendeteksi kegagalan senyap AutoTokenizer
# di bawah.
_MIN_SANE_VOCAB = 1000


def load_tokenizer(model_name: str = MODEL_NAME):
    """Muat tokenizer dan daftarkan placeholder sebagai special token.

    Returns:
        tokenizer: tokenizer dengan [URL]/[MENTION]/[NUM] terdaftar.

    Catatan: panggil `model.resize_token_embeddings(len(tokenizer))` pada model
    setelah tokenizer ini dipakai (ukuran vocab bertambah 3).

    Catatan checkpoint `indobert-lite-*`: repo Hub-nya berarsitektur ALBERT
    (`config.json` -> `AlbertModel`), sehingga `AutoTokenizer` salah menebak kelas
    `AlbertTokenizer` (berbasis SentencePiece) -- padahal repo ini HANYA menyediakan
    `vocab.txt` WordPiece (sama seperti `indobert-base-p2`), bukan file `.model`
    SentencePiece. Alih-alih error, `AlbertTokenizer` diam-diam jatuh ke vocab
    minimal 5-token (semua kata jadi `[UNK]`) -- bug senyap yang berbahaya untuk
    training. Deteksi ini lewat ukuran vocab dan fallback eksplisit ke
    `BertTokenizer` (WordPiece), yang terbukti berhasil memuat vocab penuh.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if len(tokenizer) < _MIN_SANE_VOCAB:
        print(f"[load_tokenizer] AutoTokenizer utk '{model_name}' menghasilkan vocab "
              f"mencurigakan kecil ({len(tokenizer)}) -- fallback ke BertTokenizer (WordPiece).")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        if len(tokenizer) < _MIN_SANE_VOCAB:
            raise ValueError(f"Tokenizer '{model_name}' tetap punya vocab kecil "
                             f"({len(tokenizer)}) walau sudah fallback ke BertTokenizer -- "
                             f"periksa manual repo Hub-nya.")
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tokenizer


class GamblingCommentDataset(Dataset):
    """Dataset klasifikasi biner komentar (0=non-judi, 1=promosi judi).

    Tokenisasi dilakukan on-the-fly (padding ke max_length, truncation) agar
    hemat memori dan konsisten dengan tokenizer yang dipakai model.

    Args:
        texts: iterable teks (gunakan kolom `text_clean` hasil preprocessing).
        labels: iterable label int (0/1). Boleh None untuk inferensi.
        tokenizer: hasil `load_tokenizer()`.
        max_length: panjang token maksimum (default 128).
    """

    def __init__(self, texts, labels=None, tokenizer=None, max_length: int = MAX_LENGTH):
        if tokenizer is None:
            raise ValueError("tokenizer wajib diisi (pakai load_tokenizer()).")
        self.texts = [str(t) for t in texts]
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item
