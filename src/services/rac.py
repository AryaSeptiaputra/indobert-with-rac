"""Retrieval-Augmented Classification (RAC) untuk skenario RM-c.

RM-c tidak melatih apa pun. Ia memakai ulang dua artefak milik RM-b: embedding
mean-pool dari encoder beku, dan classification head yang sudah terlatih.

Alur inferensi per sampel query:

1. `p_bert` = softmax(head(embedding_query)), yaitu cabang parametrik.
2. Cari k tetangga terdekat di indeks FAISS yang dibangun HANYA dari embedding
   TRAIN, lalu hitung distribusi label tetangga menjadi `p_retr`.
3. Fusi: `p_final = (1 - alpha) * p_bert + alpha * p_retr`, prediksi = argmax.

Fusi dilakukan pada level PROBABILITAS, dan softmax hanya diterapkan sekali di
cabang BERT sebelum fusi. Karena kedua masukan sudah berupa distribusi dan bobot
fusinya berjumlah satu, `p_final` otomatis merupakan distribusi yang sah;
softmax kedua hanya akan meratakan selisih dan bisa mengubah argmax pada kasus
nyaris seri. Fusi level probabilitas dan level logit TIDAK ekuivalen.

Indeks dibangun eksklusif dari train, dan kunci dedup NFKC-exact sudah membuang
near-duplicate lintas split, sehingga retrieval tidak bisa "curang" dengan
menemukan sampel uji di dalam indeks. Similaritas memakai cosine, diwujudkan
sebagai inner product atas vektor yang sudah dinormalisasi L2.
"""

from __future__ import annotations

import faiss
import numpy as np

from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalisasi L2 baris-per-baris agar inner product setara cosine.

    Args:
        matrix: Array (N, H).
        eps: Batas bawah norma untuk menghindari pembagian nol.

    Returns:
        Array (N, H) float32 yang kontigu dan ternormalisasi.
    """
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax numerik stabil (dikurangi nilai maksimum sebelum eksponensiasi).

    Args:
        logits: Array skor mentah.
        axis: Sumbu yang dinormalisasi.

    Returns:
        Array probabilitas dengan bentuk sama, berjumlah 1 pada `axis`.
    """
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


class RACClassifier:
    """Klasifikator RAC: indeks FAISS atas embedding train plus fusi probabilitas.

    Indeks, label train, dan hyperparameter fusi disimpan sebagai state, bukan
    dioper ulang lewat setiap pemanggilan.

    Args:
        alpha: Bobot cabang retrieval; 0 berarti RM-b murni, 1 membuang cabang
            BERT sepenuhnya.
        k: Jumlah tetangga terdekat yang diambil.
        weighting: "similarity" membobot tetangga dengan cosine yang di-clip ke
            nol; "uniform" memberi bobot sama pada tiap tetangga.
        num_labels: Jumlah kelas; `None` memakai `settings.num_labels`.

    Raises:
        ValueError: Kalau `alpha` di luar [0, 1], `k` bukan bilangan positif,
            atau `weighting` tidak dikenal.
    """

    VALID_WEIGHTINGS = ("similarity", "uniform")

    def __init__(
        self,
        alpha: float = 0.3,
        k: int = 5,
        weighting: str = "similarity",
        num_labels: int | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha harus di [0, 1]; diterima {alpha}")
        if k < 1:
            raise ValueError(f"k harus >= 1; diterima {k}")
        if weighting not in self.VALID_WEIGHTINGS:
            raise ValueError(
                f"weighting tak dikenal: {weighting!r} (harus salah satu {self.VALID_WEIGHTINGS})"
            )

        self.alpha = float(alpha)
        self.k = int(k)
        self.weighting = weighting
        self.num_labels = num_labels or settings.num_labels

        self._index: faiss.Index | None = None
        self._train_labels: np.ndarray | None = None

    @property
    def index_size(self) -> int:
        """Jumlah vektor di dalam indeks; 0 bila belum di-`fit`."""
        return 0 if self._index is None else int(self._index.ntotal)

    def fit(self, train_embeddings: np.ndarray, train_labels: np.ndarray) -> RACClassifier:
        """Bangun indeks FAISS dari embedding TRAIN.

        Indeks bertipe flat/exact, bukan aproksimasi: untuk sekitar 6.600 vektor
        biayanya sepele dan hasilnya deterministik, yang penting agar dua run
        dengan konfigurasi sama menghasilkan angka identik.

        Args:
            train_embeddings: Array (N, H) embedding split train.
            train_labels: Array (N,) label yang urutannya selaras embedding.

        Returns:
            Instance ini, agar bisa dirantai.

        Raises:
            ValueError: Kalau jumlah embedding dan label tidak sama, atau
                embedding kosong.
        """
        embeddings = l2_normalize(train_embeddings)
        labels = np.asarray(train_labels)

        if embeddings.shape[0] == 0:
            raise ValueError("embedding train kosong; indeks FAISS tidak bisa dibangun")
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"jumlah embedding ({embeddings.shape[0]}) != jumlah label ({labels.shape[0]})"
            )

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self._index = index
        self._train_labels = labels
        logger.info(
            "Indeks FAISS dibangun: %d vektor berdimensi %d",
            index.ntotal,
            embeddings.shape[1],
        )
        return self

    def retrieve(self, query_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ambil k tetangga terdekat untuk tiap query.

        Args:
            query_embeddings: Array (N, H); tidak perlu dinormalisasi lebih dulu.

        Returns:
            Tuple (similarities (N, k), indices (N, k)).

        Raises:
            RuntimeError: Kalau `fit` belum dipanggil.
            ValueError: Kalau `k` melebihi jumlah vektor di indeks. FAISS
                membalas kekurangan tetangga dengan indeks -1, dan pengindeksan
                numpy akan menerjemahkan -1 menjadi label train TERAKHIR, yaitu
                tetangga yang salah tanpa satu pun pesan galat.
        """
        if self._index is None:
            raise RuntimeError("panggil fit() sebelum retrieve()")
        if self.k > self._index.ntotal:
            raise ValueError(
                f"k={self.k} melebihi jumlah vektor di indeks ({self._index.ntotal})"
            )
        return self._index.search(l2_normalize(query_embeddings), self.k)

    def retrieval_distribution(
        self,
        similarities: np.ndarray,
        neighbor_labels: np.ndarray,
    ) -> np.ndarray:
        """Ubah label tetangga menjadi distribusi probabilitas per query.

        Args:
            similarities: Array (N, k) skor cosine tiap tetangga.
            neighbor_labels: Array (N, k) label tiap tetangga.

        Returns:
            Array (N, num_labels); tiap baris berjumlah 1. Baris yang seluruh
            bobotnya nol (semua similarity <= 0 pada mode "similarity") jatuh ke
            distribusi seragam alih-alih menghasilkan pembagian nol.
        """
        n_queries = similarities.shape[0]

        if self.weighting == "uniform":
            weights = np.ones_like(similarities, dtype=np.float32)
        else:
            weights = np.clip(similarities, 0.0, None).astype(np.float32)

        distribution = np.zeros((n_queries, self.num_labels), dtype=np.float32)
        for label in range(self.num_labels):
            mask = (neighbor_labels == label).astype(np.float32)
            distribution[:, label] = (weights * mask).sum(axis=1)

        totals = distribution.sum(axis=1, keepdims=True)
        empty_rows = totals.squeeze(-1) == 0
        if empty_rows.any():
            logger.debug(
                "%d query tanpa tetangga berbobot; memakai distribusi seragam",
                int(empty_rows.sum()),
            )
            distribution[empty_rows] = 1.0 / self.num_labels
            totals[empty_rows] = 1.0

        return distribution / totals

    def fuse(self, p_bert: np.ndarray, p_retrieval: np.ndarray) -> np.ndarray:
        """Gabungkan kedua distribusi dengan bobot `alpha`.

        Args:
            p_bert: Array (N, num_labels), softmax keluaran head.
            p_retrieval: Array (N, num_labels) hasil `retrieval_distribution`.

        Returns:
            Array (N, num_labels) distribusi hasil fusi.
        """
        return (1.0 - self.alpha) * p_bert + self.alpha * p_retrieval

    def predict(
        self,
        query_embeddings: np.ndarray,
        p_bert: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Jalankan RAC ujung ke ujung untuk satu set query.

        Args:
            query_embeddings: Array (N, H) embedding query, belum dinormalisasi.
            p_bert: Array (N, num_labels) distribusi cabang BERT, yaitu
                softmax dari logit head.

        Returns:
            Tuple (prediksi (N,), p_final (N, num_labels)).

        Raises:
            RuntimeError: Kalau `fit` belum dipanggil.
            ValueError: Kalau `k` melebihi ukuran indeks, atau bentuk `p_bert`
                tidak cocok dengan jumlah query.
        """
        if self._train_labels is None:
            raise RuntimeError("panggil fit() sebelum predict()")
        if p_bert.shape[0] != query_embeddings.shape[0]:
            raise ValueError(
                f"jumlah baris p_bert ({p_bert.shape[0]}) != jumlah query "
                f"({query_embeddings.shape[0]})"
            )

        similarities, indices = self.retrieve(query_embeddings)
        neighbor_labels = self._train_labels[indices]
        p_retrieval = self.retrieval_distribution(similarities, neighbor_labels)
        p_final = self.fuse(p_bert, p_retrieval)
        return p_final.argmax(axis=1), p_final
