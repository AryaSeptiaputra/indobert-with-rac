"""Factory encoder IndoBERT dan classification head untuk RM-a/RM-b/RM-c.

Walau ada tiga skenario, IndoBERT hanya dilatih dua kali:

- RM-a: full fine-tuning lewat `build_finetune_model`, seluruh parameter trainable.
- RM-b: encoder beku lewat `build_encoder`, hanya head (`build_head`) yang dilatih.
- RM-c: memakai ulang encoder beku dan head RM-b, tanpa training baru.

Placeholder [URL]/[MENTION]/[NUM] menambah baris embedding, sehingga model wajib
`resize_token_embeddings`. Untuk encoder beku, baris baru itu tidak akan pernah
dilatih, jadi diinisialisasi dengan rata-rata embedding subword kata deskriptif
agar berisi representasi bermakna, bukan noise acak.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.config import (
    MENTION_PLACEHOLDER,
    NUM_PLACEHOLDER,
    URL_PLACEHOLDER,
    settings,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BERT_BASE_HIDDEN_SIZE = 768

# Kata bahasa Indonesia yang dikenal vocab IndoBERT, dipakai sebagai benih
# embedding tiap placeholder.
SEED_WORDS: dict[str, str] = {
    URL_PLACEHOLDER: "tautan",
    MENTION_PLACEHOLDER: "akun",
    NUM_PLACEHOLDER: "angka",
}


@torch.no_grad()
def init_special_token_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    seed_words: dict[str, str] | None = None,
) -> int:
    """Isi embedding special token dengan rata-rata embedding subword kata benih.

    Harus dipanggil SETELAH `resize_token_embeddings`. Aman untuk model dengan
    maupun tanpa classification head.

    Args:
        model: Model yang embedding input-nya akan diubah.
        tokenizer: Tokenizer yang special token-nya sudah terdaftar.
        seed_words: Peta placeholder ke kata benih; `None` memakai `SEED_WORDS`.

    Returns:
        Jumlah placeholder yang berhasil diinisialisasi.
    """
    seed_words = seed_words or SEED_WORDS
    embeddings = model.get_input_embeddings()
    initialised = 0

    for token, word in seed_words.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            logger.warning(
                "Special token %s tidak ada di vocab (dipetakan ke UNK); "
                "embedding-nya dibiarkan acak",
                token,
            )
            continue
        piece_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if not piece_ids:
            logger.warning(
                "Kata benih %r untuk %s tidak menghasilkan subword; "
                "embedding token itu dibiarkan acak",
                word,
                token,
            )
            continue
        embeddings.weight[token_id] = embeddings.weight[piece_ids].mean(dim=0)
        initialised += 1

    logger.info(
        "Inisialisasi embedding special token: %d/%d", initialised, len(seed_words)
    )
    return initialised


def build_finetune_model(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str | None = None,
    num_labels: int | None = None,
) -> PreTrainedModel:
    """RM-a: IndoBERT + classification head, seluruh parameter trainable.

    Args:
        tokenizer: Hasil `load_tokenizer()`.
        model_name: Nama repo Hub; `None` memakai `settings.base_model`.
        num_labels: Jumlah kelas; `None` memakai `settings.num_labels`.

    Returns:
        Model siap latih dengan embedding sudah di-resize dan diinisialisasi.

    Raises:
        OSError: Kalau bobot pra-latih tidak bisa diunduh.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name or settings.base_model,
        num_labels=num_labels or settings.num_labels,
    )
    _resize_and_init(model, tokenizer)
    model.requires_grad_(True)
    return model


def build_encoder(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str | None = None,
) -> PreTrainedModel:
    """RM-b/RM-c: encoder IndoBERT BEKU (requires_grad False, mode eval).

    Args:
        tokenizer: Hasil `load_tokenizer()`.
        model_name: Nama repo Hub; `None` memakai `settings.base_model`.

    Returns:
        Encoder beku siap dipakai untuk ekstraksi fitur.

    Raises:
        OSError: Kalau bobot pra-latih tidak bisa diunduh.
    """
    encoder = AutoModel.from_pretrained(model_name or settings.base_model)
    _resize_and_init(encoder, tokenizer)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked mean-pooling atas dimensi token.

    Token padding diabaikan lewat `attention_mask`. Representasi kalimat yang
    sama dipakai untuk head RM-b dan indeks FAISS RM-c agar keduanya konsisten.

    Args:
        last_hidden_state: Keluaran encoder, bentuk (B, T, H).
        attention_mask: Mask token nyata, bentuk (B, T).

    Returns:
        Tensor (B, H) hasil rata-rata terbobot mask.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class LinearHead(nn.Module):
    """Head RM-b varian linear: Dropout lalu Linear (linear probe).

    Trainable params sekitar (H+1) * num_labels, jauh di bawah 110 juta milik RM-a.
    """

    def __init__(
        self,
        hidden_size: int = BERT_BASE_HIDDEN_SIZE,
        num_labels: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels or settings.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))


class MLPHead(nn.Module):
    """Head RM-b varian MLP: Linear -> ReLU -> Dropout -> Linear.

    Menguji apakah kapasitas tambahan satu hidden layer menutup selisih terhadap
    RM-a. Trainable params sekitar H*hidden_dim + hidden_dim*num_labels.
    """

    def __init__(
        self,
        hidden_size: int = BERT_BASE_HIDDEN_SIZE,
        hidden_dim: int = 256,
        num_labels: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels or settings.num_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def build_head(
    arch: str = "linear",
    hidden_size: int = BERT_BASE_HIDDEN_SIZE,
    num_labels: int | None = None,
    dropout: float = 0.1,
    hidden_dim: int = 256,
) -> nn.Module:
    """Factory head RM-b.

    Args:
        arch: "linear" untuk `LinearHead`, "mlp" untuk `MLPHead`.
        hidden_size: Dimensi fitur masukan (dimensi hidden encoder).
        num_labels: Jumlah kelas; `None` memakai `settings.num_labels`.
        dropout: Probabilitas dropout.
        hidden_dim: Lebar hidden layer, hanya dipakai oleh "mlp".

    Returns:
        Modul head yang belum dilatih.

    Raises:
        ValueError: Kalau `arch` bukan "linear" atau "mlp".
    """
    if arch == "linear":
        return LinearHead(
            hidden_size=hidden_size, num_labels=num_labels, dropout=dropout
        )
    if arch == "mlp":
        return MLPHead(
            hidden_size=hidden_size,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
    raise ValueError(f"arsitektur head tak dikenal: {arch!r} (harus 'linear' atau 'mlp')")


def _resize_and_init(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> None:
    model.resize_token_embeddings(len(tokenizer))
    init_special_token_embeddings(model, tokenizer)
