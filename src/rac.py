"""Retrieval-Augmented Classification (RAC) untuk RM-c.

RM-c TIDAK melatih IndoBERT. Ia memakai ulang, dari RM-b:
- embedding beku (mean-pool) train/val/test  -> results/features/*.npy
- head klasifikasi terlatih                  -> results/checkpoints/rmb/head_best.pt

Alur inferензi RAC per sampel query:
    1. p_bert  = softmax(head(embedding_query))            (cabang BERT)
    2. cari k tetangga terdekat di index FAISS (dibangun HANYA dari embedding
       TRAIN -> anti-leakage), hitung distribusi label tetangga -> p_retr
    3. fusi:  p_final = (1 - alpha) * p_bert + alpha * p_retr
       prediksi = argmax(p_final)

Kunci dedup NFKC-exact + FAISS-train-only mencegah kebocoran (near-duplicate
train<->test yang akan membuat retrieval "curang"). Lihat EDA_PLAN.md prinsip #2.

Similaritas = cosine (embedding di-L2-normalisasi -> inner product).
"""

from __future__ import annotations

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except ImportError:  # faiss opsional; beri pesan jelas saat dipakai
    _HAS_FAISS = False


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalisasi L2 baris-per-baris (untuk cosine via inner product)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def build_faiss_index(train_emb: np.ndarray):
    """Bangun index FAISS IndexFlatIP dari embedding TRAIN (dinormalisasi).

    Flat/exact (bukan approx) -> cocok utk ~6,6k vektor & hasil deterministik.
    Mengembalikan (index, train_emb_ternormalisasi).
    """
    if not _HAS_FAISS:
        raise ImportError("faiss belum terpasang. Jalankan: pip install faiss-cpu")
    emb = l2_normalize(train_emb)
    index = faiss.IndexFlatIP(emb.shape[1])   # inner product = cosine (sudah dinormalisasi)
    index.add(emb)
    return index, emb


def retrieve(index, query_emb: np.ndarray, k: int):
    """k tetangga terdekat. Mengembalikan (sims (N,k), idx (N,k))."""
    q = l2_normalize(query_emb)
    sims, idx = index.search(q, k)
    return sims, idx


def retrieval_distribution(sims: np.ndarray, neighbor_labels: np.ndarray,
                           num_labels: int = 2, weighting: str = "similarity") -> np.ndarray:
    """Distribusi label dari k tetangga -> (N, num_labels), tiap baris berjumlah 1.

    weighting:
      - 'similarity' (default): bobot = max(cosine, 0), tetangga lebih mirip lebih
        berpengaruh; jika semua 0 -> fallback uniform.
      - 'uniform': tiap tetangga berbobot sama (voting mayoritas halus).
    """
    N, k = sims.shape
    if weighting == "uniform":
        w = np.ones_like(sims, dtype=np.float32)
    elif weighting == "similarity":
        w = np.clip(sims, 0.0, None).astype(np.float32)
    else:
        raise ValueError(f"weighting tak dikenal: {weighting}")

    dist = np.zeros((N, num_labels), dtype=np.float32)
    for c in range(num_labels):
        mask = (neighbor_labels == c).astype(np.float32)
        dist[:, c] = (w * mask).sum(axis=1)

    total = dist.sum(axis=1, keepdims=True)
    # baris dgn total 0 (mis. semua sim<=0) -> uniform
    zero = (total.squeeze(-1) == 0)
    if zero.any():
        dist[zero] = 1.0 / num_labels
        total[zero] = 1.0
    return dist / total


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fuse(p_bert: np.ndarray, p_retr: np.ndarray, alpha: float) -> np.ndarray:
    """Fusi distribusi probabilitas: (1-alpha)*p_bert + alpha*p_retr."""
    return (1.0 - alpha) * p_bert + alpha * p_retr


def rac_predict(index, train_labels: np.ndarray, query_emb: np.ndarray,
                p_bert: np.ndarray, k: int = 5, alpha: float = 0.3,
                weighting: str = "similarity"):
    """Prediksi RAC end-to-end untuk satu set query.

    Args:
        index       : FAISS index dari build_faiss_index.
        train_labels: label train (selaras urutan index).
        query_emb   : embedding query (N, H) — belum dinormalisasi.
        p_bert      : distribusi cabang BERT (N, num_labels) = softmax(head_logits).
        k, alpha    : hyperparameter retrieval & bobot fusi.
    Returns:
        preds (N,), p_final (N, num_labels)
    """
    sims, idx = retrieve(index, query_emb, k)
    neigh_labels = np.asarray(train_labels)[idx]
    p_retr = retrieval_distribution(sims, neigh_labels, num_labels=p_bert.shape[1],
                                    weighting=weighting)
    p_final = fuse(p_bert, p_retr, alpha)
    return p_final.argmax(axis=1), p_final
