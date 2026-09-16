"""Ablasi rumus fusi RAC (RM-c): bandingkan mekanisme fusi alternatif dari literatur.

Modul ini TIDAK mengubah pipeline produksi RM-c (`src.services.rac.RACClassifier.fuse`,
fusi linear-konveks tetap `p_final = (1-alpha)*p_bert + alpha*p_retrieval`). Ia memakai
ulang retrieval FAISS milik `RACClassifier` (indeks dari split train saja) lalu menguji
empat rumus fusi alternatif di atas `p_bert`/`p_retrieval`/logit yang sama, supaya
kontribusi mekanisme fusi itu sendiri bisa diisolasi dari kontribusi sinyal retrieval.

Rumus 1 (Long dkk., 2022) beroperasi di level SKOR (logit mentah + vote tertimbang
sebelum dinormalisasi), bukan level probabilitas -- sengaja menyimpang dari aturan
"softmax sekali, sebelum fusi" milik `rac.py` sebagai bagian dari perbandingan
literatur, dan tidak pernah memanggil `RACClassifier.fuse`/`predict`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.services.evaluation import ClassificationEvaluator
from src.services.features import FeatureSet
from src.services.rac import RACClassifier, l2_normalize, softmax
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

FusionFormula = Literal["rumus1", "rumus2", "rumus3", "rumus4"]

# Penambah kecil sebelum pemangkatan di Rumus 4, agar 0**alpha (alpha < 1) tidak
# menghasilkan NaN pada baris dengan probabilitas nol persis.
POW_EPS = 1e-12


@dataclass(frozen=True)
class FusionFormulaConfig:
    """Satu konfigurasi rumus fusi yang diuji.

    Attributes:
        formula: Nama rumus, "rumus1".."rumus4".
        alpha: Bobot cabang retrieval untuk Rumus 4; rumus lain mengabaikannya
            (Rumus 1 tidak punya alpha, Rumus 2/3 menghitungnya secara adaptif).
        k: Jumlah tetangga terdekat, sekadar dicatat di baris hasil -- retrieval
            aktualnya ditentukan oleh `FusionFormulaComparator.k`.
        weighting: Skema bobot tetangga, sama seperti `RACClassifier`.
    """

    formula: FusionFormula
    alpha: float | None = None
    k: int = 5
    weighting: Literal["similarity", "uniform"] = "similarity"


class FusionFormulaComparator:
    """Bandingkan rumus fusi RAC alternatif di atas retrieval & head RM-b yang sama.

    Retrieval (k, weighting) ditahan tetap lintas rumus agar perbandingan hanya
    mengukur pengaruh mekanisme fusi, bukan pengaruh hyperparameter retrieval.

    Args:
        features: Embedding beku ketiga split, sama seperti dipakai RM-c produksi.
        head: Head RM-b terbaik yang sudah dilatih (tidak pernah diubah di sini).
        device: Device komputasi.
        k: Jumlah tetangga terdekat.
        weighting: Skema bobot tetangga.
    """

    def __init__(
        self,
        features: FeatureSet,
        head: nn.Module,
        device: torch.device,
        k: int = 5,
        weighting: Literal["similarity", "uniform"] = "similarity",
    ) -> None:
        self.features = features
        self.head = head
        self.device = device
        self.k = k
        self.weighting = weighting
        self.evaluator = ClassificationEvaluator()
        self._retrieval: RACClassifier | None = None

    def _fit_retrieval(self) -> RACClassifier:
        """Bangun indeks FAISS dari train sekali; dipakai ulang oleh semua rumus.

        `alpha=0.0` di sini hanya untuk lolos validasi konstruktor -- `.fuse()` dan
        `.predict()` milik instance ini tidak pernah dipanggil, hanya
        `.fit()`/`.retrieve()`/`.retrieval_distribution()`.
        """
        if self._retrieval is None:
            train_emb, train_lab = self.features["train"]
            self._retrieval = RACClassifier(alpha=0.0, k=self.k, weighting=self.weighting).fit(
                train_emb, train_lab
            )
        return self._retrieval

    def _p_bert_and_logits(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Forward head sekali untuk satu split; kembalikan (p_bert, logit mentah)."""
        query_emb, _ = self.features[split]
        self.head.eval().to(self.device)
        with torch.no_grad():
            logits = self.head(torch.tensor(query_emb, device=self.device))
        logits_np = logits.cpu().numpy()
        return softmax(logits_np), logits_np

    def raw_retrieval_scores(
        self,
        similarities: np.ndarray,
        neighbor_labels: np.ndarray,
        num_labels: int,
    ) -> np.ndarray:
        """Vote tertimbang per kelas SEBELUM dinormalisasi menjadi distribusi.

        Logika bobot sama persis dengan `RACClassifier.retrieval_distribution`
        (bobot = clip(similarity, 0, None) per tetangga untuk "similarity", atau 1
        untuk "uniform", dijumlah per label), tapi tanpa langkah pembagian baris
        terakhir -- dibutuhkan Rumus 1 yang fusinya di level skor, bukan distribusi.
        Duplikasi kecil ini disengaja agar kontrak publik `RACClassifier` (dan
        test-nya yang sudah ada) tidak perlu berubah demi kebutuhan ablasi ini.

        Args:
            similarities: Array (N, k) skor cosine tiap tetangga.
            neighbor_labels: Array (N, k) label tiap tetangga.
            num_labels: Jumlah kelas.

        Returns:
            Array (N, num_labels); TIDAK dijamin berjumlah 1 per baris.
        """
        if self.weighting == "uniform":
            weights = np.ones_like(similarities, dtype=np.float32)
        else:
            weights = np.clip(similarities, 0.0, None).astype(np.float32)

        raw = np.zeros((similarities.shape[0], num_labels), dtype=np.float32)
        for label in range(num_labels):
            mask = (neighbor_labels == label).astype(np.float32)
            raw[:, label] = (weights * mask).sum(axis=1)
        return raw

    def rumus1_score_fusion(self, logits: np.ndarray, raw_retrieval: np.ndarray) -> np.ndarray:
        """Skema asli RAC (Long dkk., 2022): rata-rata dua skor yang di-L2-normalisasi.

        `f(x) = (L/2) * (f_retrieval/||f_retrieval||_2 + f_head/||f_head||_2)`. Fusi
        di level SKOR, bukan probabilitas: `f(x)` bukan distribusi yang sah (bisa
        memuat nilai negatif, tidak dijamin berjumlah 1). Prediksinya `argmax(f(x))`.

        Args:
            logits: Array (N, L) logit mentah head (pre-softmax).
            raw_retrieval: Array (N, L) vote tertimbang belum dinormalisasi, dari
                `raw_retrieval_scores`.

        Returns:
            Array (N, L) skor gabungan `f(x)`.
        """
        num_labels = logits.shape[1]
        return (num_labels / 2.0) * (l2_normalize(raw_retrieval) + l2_normalize(logits))

    def rumus2_similarity_alpha(
        self,
        p_bert: np.ndarray,
        similarities: np.ndarray,
        p_retrieval: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alpha adaptif = rata-rata similarity k tetangga teratas (parameter-free).

        Args:
            p_bert: Array (N, L) softmax keluaran head.
            similarities: Array (N, k) skor cosine tiap tetangga.
            p_retrieval: Array (N, L) distribusi retrieval.

        Returns:
            Tuple (p_final (N, L), alpha (N,)).
        """
        alpha = np.clip(similarities.mean(axis=1), 0.0, 1.0).astype(np.float32)
        alpha_col = alpha[:, None]
        p_final = (1.0 - alpha_col) * p_bert + alpha_col * p_retrieval
        return p_final, alpha

    def rumus3_uncertainty_alpha(
        self,
        p_bert: np.ndarray,
        p_retrieval: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alpha adaptif = ketidakpastian head, `1 - max(p_head)` (parameter-free).

        Args:
            p_bert: Array (N, L) softmax keluaran head.
            p_retrieval: Array (N, L) distribusi retrieval.

        Returns:
            Tuple (p_final (N, L), alpha (N,)).
        """
        alpha = (1.0 - p_bert.max(axis=1)).astype(np.float32)
        alpha_col = alpha[:, None]
        p_final = (1.0 - alpha_col) * p_bert + alpha_col * p_retrieval
        return p_final, alpha

    def rumus4_geometric_pool(
        self,
        p_bert: np.ndarray,
        p_retrieval: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Geometric pooling / product-of-experts: `p_head^(1-alpha) * p_retr^alpha`.

        Args:
            p_bert: Array (N, L) softmax keluaran head.
            p_retrieval: Array (N, L) distribusi retrieval.
            alpha: Bobot cabang retrieval, tetap (bukan adaptif per sampel).

        Returns:
            Array (N, L) distribusi hasil fusi, dinormalisasi ulang per baris.
        """
        pooled = np.power(p_bert + POW_EPS, 1.0 - alpha) * np.power(p_retrieval + POW_EPS, alpha)
        totals = pooled.sum(axis=1, keepdims=True)
        return pooled / totals

    def evaluate(
        self,
        config: FusionFormulaConfig,
        split: str = "val",
    ) -> tuple[dict[str, float | int], dict[str, object]]:
        """Evaluasi satu rumus/alpha pada satu split.

        Args:
            config: Rumus dan hyperparameter yang diuji.
            split: Split yang dievaluasi; harus "val" selama ablasi (split test
                hanya dibuka di 05_final_benchmark.ipynb).

        Returns:
            Tuple (metrik via `ClassificationEvaluator`, extras). `extras` memuat
            `eval_time_s`, `index_vectors`, `preds`, dan `alpha_used` (`None` untuk
            Rumus 1 yang tidak punya alpha).

        Raises:
            ValueError: Kalau `config.formula == "rumus4"` tapi `config.alpha`
                kosong, atau `config.formula` tidak dikenal.
        """
        retrieval = self._fit_retrieval()
        p_bert, logits = self._p_bert_and_logits(split)
        query_emb, query_lab = self.features[split]
        _, train_lab = self.features["train"]

        similarities, indices = retrieval.retrieve(query_emb)
        neighbor_labels = train_lab[indices]
        num_labels = p_bert.shape[1]
        p_retrieval = retrieval.retrieval_distribution(similarities, neighbor_labels)

        started = time.perf_counter()
        alpha_used: np.ndarray | float | None = None
        if config.formula == "rumus1":
            raw_retrieval = self.raw_retrieval_scores(similarities, neighbor_labels, num_labels)
            scores = self.rumus1_score_fusion(logits, raw_retrieval)
            predictions = scores.argmax(axis=1)
        elif config.formula == "rumus2":
            p_final, alpha_used = self.rumus2_similarity_alpha(p_bert, similarities, p_retrieval)
            predictions = p_final.argmax(axis=1)
        elif config.formula == "rumus3":
            p_final, alpha_used = self.rumus3_uncertainty_alpha(p_bert, p_retrieval)
            predictions = p_final.argmax(axis=1)
        elif config.formula == "rumus4":
            if config.alpha is None:
                raise ValueError("rumus4 butuh config.alpha")
            p_final = self.rumus4_geometric_pool(p_bert, p_retrieval, config.alpha)
            predictions = p_final.argmax(axis=1)
            alpha_used = config.alpha
        else:
            raise ValueError(f"rumus tak dikenal: {config.formula!r}")
        elapsed = round(time.perf_counter() - started, 4)

        extras: dict[str, object] = {
            "eval_time_s": elapsed,
            "index_vectors": retrieval.index_size,
            "preds": predictions,
            "alpha_used": alpha_used,
        }
        return self.evaluator.metrics(query_lab, predictions), extras

    def run_all(
        self,
        split: str = "val",
        rumus4_alphas: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    ) -> pd.DataFrame:
        """Jalankan keempat rumus (Rumus 4 di-sweep) dan kumpulkan hasilnya.

        Args:
            split: Split yang dievaluasi.
            rumus4_alphas: Grid alpha khusus Rumus 4; rumus lain tidak butuh alpha
                eksternal.

        Returns:
            DataFrame satu baris per rumus (Rumus 4: satu baris per alpha), kolom
            `formula, alpha, k, weighting, val_f1_macro, val_f1_judi, val_accuracy,
            eval_time_s, index_vectors`.
        """
        rows: list[dict[str, object]] = []
        for formula in ("rumus1", "rumus2", "rumus3"):
            config = FusionFormulaConfig(formula=formula, k=self.k, weighting=self.weighting)
            metrics, extras = self.evaluate(config, split=split)
            rows.append(self._row(config, metrics, extras))
            logger.info("[%s] val F1-macro %.4f", formula, metrics["f1_macro"])

        for alpha in rumus4_alphas:
            config = FusionFormulaConfig(
                formula="rumus4", alpha=alpha, k=self.k, weighting=self.weighting
            )
            metrics, extras = self.evaluate(config, split=split)
            rows.append(self._row(config, metrics, extras))
            logger.info("[rumus4 alpha=%.2f] val F1-macro %.4f", alpha, metrics["f1_macro"])

        return pd.DataFrame(rows)

    @staticmethod
    def _row(
        config: FusionFormulaConfig,
        metrics: dict[str, float | int],
        extras: dict[str, object],
    ) -> dict[str, object]:
        return {
            "formula": config.formula,
            "alpha": config.alpha,
            "k": config.k,
            "weighting": config.weighting,
            "val_f1_macro": metrics["f1_macro"],
            "val_f1_judi": metrics["f1_class1"],
            "val_accuracy": metrics["accuracy"],
            "eval_time_s": extras["eval_time_s"],
            "index_vectors": extras["index_vectors"],
        }
