"""Pembangun model & ekstraksi fitur IndoBERT untuk RM-a/RM-b/RM-c.

Dua jalur adaptasi (walau 3 skenario, IndoBERT hanya dilatih 2x):
- RM-a  : full fine-tuning  -> build_finetune_model() (semua param trainable).
- RM-b  : frozen encoder    -> build_encoder() + extract_features() + FrozenHead.
- RM-c  : reuse encoder beku + fitur train RM-b + FAISS (tanpa training baru).

Catatan special token: [URL]/[MENTION]/[NUM] ditambahkan ke tokenizer saat
preprocessing. Karena menambah baris embedding, model WAJIB
`resize_token_embeddings(len(tokenizer))`. Untuk encoder beku (RM-b/RM-c) baris
baru itu tidak pernah dilatih -> di-`init_special_token_embeddings` agar berisi
representasi bermakna (rata-rata subword kata deskriptif), bukan noise acak.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification

from dataset import MODEL_NAME, SPECIAL_TOKENS

# Kata deskriptif untuk inisialisasi embedding tiap special token (bahasa Indonesia,
# dikenal vocab IndoBERT). Dipakai oleh init_special_token_embeddings.
_SEED_WORDS = {
    "[URL]": "tautan",
    "[MENTION]": "akun",
    "[NUM]": "angka",
}


@torch.no_grad()
def init_special_token_embeddings(model, tokenizer, seed_words: dict | None = None):
    """Inisialisasi embedding special token = rata-rata embedding subword kata seed.

    Dipanggil SETELAH resize_token_embeddings. Aman untuk model dgn/ tanpa head.
    """
    seed_words = seed_words or _SEED_WORDS
    emb = model.get_input_embeddings()  # nn.Embedding (vocab, hidden)
    for tok, word in seed_words.items():
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            continue
        piece_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if not piece_ids:
            continue
        vec = emb.weight[piece_ids].mean(dim=0)
        emb.weight[tok_id] = vec


def _resize_and_init(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    init_special_token_embeddings(model, tokenizer)
    return model


def build_finetune_model(tokenizer, num_labels: int = 2, model_name: str = MODEL_NAME):
    """RM-a: IndoBERT + head klasifikasi, SEMUA parameter trainable."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )
    _resize_and_init(model, tokenizer)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def build_encoder(tokenizer, model_name: str = MODEL_NAME):
    """RM-b/RM-c: encoder IndoBERT BEKU (requires_grad=False, eval mode)."""
    encoder = AutoModel.from_pretrained(model_name)
    _resize_and_init(encoder, tokenizer)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean-pooling atas dimensi token -> (B, H).

    Mengabaikan token padding via attention_mask. Dipakai sebagai representasi
    kalimat untuk head RM-b dan index FAISS RM-c (konsisten).
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                   # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                         # (B, 1)
    return summed / counts


@torch.no_grad()
def extract_features(encoder, dataloader, device, use_amp: bool = True):
    """Ekstrak embedding mean-pool dari encoder beku untuk seluruh dataloader.

    Returns:
        embeddings: np.ndarray (N, H) float32
        labels    : np.ndarray (N,) int64  (None bila batch tak punya 'labels')
    """
    encoder.eval()
    feats, labels = [], []
    has_labels = True
    amp_enabled = use_amp and device.type == "cuda"
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        kwargs = {"input_ids": input_ids, "attention_mask": attn}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out = encoder(**kwargs)
        pooled = mean_pool(out.last_hidden_state, attn).float().cpu().numpy()
        feats.append(pooled)
        if "labels" in batch:
            labels.append(batch["labels"].numpy())
        else:
            has_labels = False
    embeddings = np.concatenate(feats, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.int64) if has_labels else None
    return embeddings, labels


class FrozenHead(nn.Module):
    """Classification head RM-b: Dropout + Linear (linear probe di atas fitur beku).

    Trainable params ~ (H+1)*num_labels; kontras dgn ~125M param RM-a.
    """

    def __init__(self, hidden_size: int = 768, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))
