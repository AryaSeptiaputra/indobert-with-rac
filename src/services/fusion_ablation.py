"""Rumus fusi RAC (RM-c): fusi linear produksi dan empat rumus alternatif dari literatur.

Modul ini TIDAK mengubah pipeline produksi RM-c (`src.services.rac.RACClassifier.fuse`,
fusi linear-konveks tetap `p_final = (1-alpha)*p_bert + alpha*p_retrieval`). Rumus
`linear` di sini memanggil `RACClassifier.fuse` apa adanya, sedangkan empat rumus
lain diuji di atas `p_bert`/`p_retrieval`/logit dan retrieval yang sama, supaya
kontribusi mekanisme fusi itu sendiri bisa diisolasi dari kontribusi sinyal retrieval.

Rumus 1 (Long dkk., 2022) beroperasi di level SKOR (logit mentah + vote tertimbang
sebelum dinormalisasi), bukan level probabilitas -- sengaja menyimpang dari aturan
"softmax sekali, sebelum fusi" milik `rac.py` sebagai bagian dari perbandingan
literatur, dan tidak pernah memanggil `RACClassifier.fuse`/`predict`.

Retrieval (k, weighting) mengikuti `FusionFormulaConfig`, dan hasil pencarian
tetangga dibagi lewat `NeighborCache`, sehingga menyapu banyak head dan banyak
konfigurasi tidak mengulang pencarian FAISS.
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
from src.services.rac import NeighborCache, RACClassifier, l2_normalize, softmax
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

FusionFormula = Literal["linear", "rumus1", "rumus2", "rumus3", "rumus4"]

# Penambah kecil sebelum pemangkatan di Rumus 4, agar 0**alpha (alpha < 1) tidak
# menghasilkan NaN pada baris dengan probabilitas nol persis.
POW_EPS = 1e-12


@dataclass(frozen=True)
class FusionFormulaConfig:
    """Satu konfigurasi rumus fusi yang diuji.

    Attributes:
        formula: Nama rumus, "linear" atau "rumus1".."rumus4".
        alpha: Bobot cabang retrieval untuk `linear` dan Rumus 4; Rumus 1 tidak
            punya alpha, Rumus 2/3 menghitungnya secara adaptif per sampel.
        k: Jumlah tetangga terdekat yang dipakai retrieval.
        weighting: Skema bobot tetangga, sama seperti `RACClassifier`.
    """

    formula: FusionFormula
    alpha: float | None = None
    k: int = 5
    weighting: Literal["similarity", "uniform"] = "similarity"


class FusionFormulaComparator:
    """Evaluasi rumus fusi RAC di atas satu head RM-b dan retrieval train-only.

    Head tidak pernah diubah. Logit head per split dihitung sekali lalu disimpan,
    dan pencarian tetangga dibagi lewat `NeighborCache` (boleh dipakai bersama
    oleh banyak comparator, misalnya satu per head RM-b).

    Args:
        features: Embedding beku ketiga split, sama seperti dipakai RM-c produksi.
        head: Head RM-b yang sudah dilatih.
        device: Device komputasi.
        k: k terbesar yang boleh diminta konfigurasi, sekaligus k bawaan `run_all`.
            Diabaikan bila `retrieval` diberikan.
        weighting: Skema bobot bawaan `run_all`.
        retrieval: Cache tetangga bersama; `None` membangun sendiri dengan `max_k=k`.
    """

    def __init__(
        self,
        features: FeatureSet,
        head: nn.Module,
        device: torch.device,
        k: int = 5,
        weighting: Literal["similarity", "uniform"] = "similarity",
        retrieval: NeighborCache | None = None,
    ) -> None:
        self.features = features
        self.head = head
        self.device = device
        self.k = k
        self.weighting = weighting
        self.evaluator = ClassificationEvaluator()
        self._retrieval = retrieval
        self._logits: dict[str, np.ndarray] = {}

    def _fit_retrieval(self) -> NeighborCache:
        """Bangun cache tetangga dari train sekali; dipakai ulang oleh semua rumus."""
        if self._retrieval is None:
            train_emb, train_lab = self.features["train"]
            self._retrieval = NeighborCache(train_emb, train_lab, max_k=self.k)
        return self._retrieval

    def _head_logits(self, split: str) -> np.ndarray:
        """Forward head sekali per split; kembalikan logit mentah (N, L)."""
        if split not in self._logits:
            query_emb, _ = self.features[split]
            self.head.eval().to(self.device)
            with torch.no_grad():
                logits = self.head(torch.tensor(query_emb, device=self.device))
            self._logits[split] = logits.cpu().numpy()
        return self._logits[split]

    def raw_retrieval_scores(
        self,
        similarities: np.ndarray,
        neighbor_labels: np.ndarray,
        num_labels: int,
        weighting: str | None = None,
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
            weighting: Skema bobot; `None` memakai `self.weighting`.

        Returns:
            Array (N, num_labels); TIDAK dijamin berjumlah 1 per baris.
        """
        weighting = weighting or self.weighting
        if weighting == "uniform":
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

    def fuse(
        self,
        config: FusionFormulaConfig,
        logits: np.ndarray,
        similarities: np.ndarray,
        neighbor_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | float | None]:
        """Gabungkan cabang head dan cabang retrieval dengan satu rumus.

        Dipakai bersama oleh evaluasi batch dan benchmark inferensi satu sampel,
        supaya keduanya menjalankan rumus yang persis sama.

        Args:
            config: Rumus dan hyperparameter-nya.
            logits: Array (N, L) logit mentah head.
            similarities: Array (N, k) skor cosine tiap tetangga.
            neighbor_labels: Array (N, k) label tiap tetangga.

        Returns:
            Tuple (skor (N, L), alpha yang dipakai). Skor adalah distribusi sah
            untuk semua rumus KECUALI Rumus 1; prediksinya `argmax` skor. Alpha
            berupa array (N,) untuk Rumus 2/3, float untuk `linear`/Rumus 4, dan
            `None` untuk Rumus 1.

        Raises:
            ValueError: Kalau `linear` atau Rumus 4 tanpa `config.alpha`, atau
                rumus tidak dikenal.
        """
        num_labels = logits.shape[1]
        p_bert = softmax(logits)

        if config.formula == "rumus1":
            raw = self.raw_retrieval_scores(
                similarities, neighbor_labels, num_labels, config.weighting
            )
            return self.rumus1_score_fusion(logits, raw), None

        p_retrieval = RACClassifier(
            k=config.k, weighting=config.weighting, num_labels=num_labels
        ).retrieval_distribution(similarities, neighbor_labels)

        if config.formula == "linear":
            if config.alpha is None:
                raise ValueError("linear butuh config.alpha")
            fuser = RACClassifier(
                alpha=config.alpha, k=config.k, weighting=config.weighting, num_labels=num_labels
            )
            return fuser.fuse(p_bert, p_retrieval), config.alpha
        if config.formula == "rumus2":
            return self.rumus2_similarity_alpha(p_bert, similarities, p_retrieval)
        if config.formula == "rumus3":
            return self.rumus3_uncertainty_alpha(p_bert, p_retrieval)
        if config.formula == "rumus4":
            if config.alpha is None:
                raise ValueError("rumus4 butuh config.alpha")
            return self.rumus4_geometric_pool(p_bert, p_retrieval, config.alpha), config.alpha
        raise ValueError(f"rumus tak dikenal: {config.formula!r}")

    @staticmethod
    def to_positive_score(formula: str, scores: np.ndarray) -> np.ndarray:
        """Skor kelas positif (judi) untuk kurva PR/ROC.

        Rumus 1 tidak menghasilkan probabilitas, jadi skornya dilewatkan softmax.
        Untuk dua kelas transformasi itu monoton terhadap selisih skor, sehingga
        urutan sampel dan kurva PR/ROC tidak berubah.

        Args:
            formula: Nama rumus.
            scores: Array (N, L) keluaran `fuse`.

        Returns:
            Array (N,) skor kelas indeks 1.
        """
        if formula == "rumus1":
            return softmax(scores)[:, 1]
        return scores[:, 1]

    def evaluate(
        self,
        config: FusionFormulaConfig,
        split: str = "val",
    ) -> tuple[dict[str, float | int], dict[str, object]]:
        """Evaluasi satu rumus/alpha/k pada satu split.

        Args:
            config: Rumus dan hyperparameter yang diuji.
            split: Split yang dievaluasi; "val" selama eksplorasi (split test
                hanya dibuka di 05_final_benchmark.ipynb).

        Returns:
            Tuple (metrik via `ClassificationEvaluator`, extras). `extras` memuat
            `eval_time_s`, `index_vectors`, `preds`, `p_judi`, dan `alpha_used`
            (`None` untuk Rumus 1 yang tidak punya alpha).

        Raises:
            ValueError: Kalau alpha yang dibutuhkan kosong, rumus tak dikenal,
                atau `config.k` melebihi k maksimum cache.
        """
        retrieval = self._fit_retrieval()
        logits = self._head_logits(split)
        query_emb, query_lab = self.features[split]
        similarities, neighbor_labels = retrieval.neighbors(split, query_emb, config.k)

        started = time.perf_counter()
        scores, alpha_used = self.fuse(config, logits, similarities, neighbor_labels)
        predictions = scores.argmax(axis=1)
        elapsed = round(time.perf_counter() - started, 4)

        extras: dict[str, object] = {
            "eval_time_s": elapsed,
            "index_vectors": retrieval.index_size,
            "preds": predictions,
            "p_judi": self.to_positive_score(config.formula, scores),
            "alpha_used": alpha_used,
        }
        return self.evaluator.metrics(query_lab, predictions), extras

    def run_all(
        self,
        split: str = "val",
        rumus4_alphas: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
        linear_alpha: float | None = None,
    ) -> pd.DataFrame:
        """Jalankan rumus-rumus di k dan weighting bawaan (Rumus 4 di-sweep).

        Args:
            split: Split yang dievaluasi.
            rumus4_alphas: Grid alpha khusus Rumus 4; rumus lain tidak butuh alpha
                eksternal.
            linear_alpha: Bila diberikan, fusi `linear` produksi ikut dijalankan
                dengan alpha ini sebagai baris pembanding.

        Returns:
            DataFrame satu baris per rumus (Rumus 4: satu baris per alpha), kolom
            `formula, alpha, k, weighting, val_f1_macro, val_f1_judi, val_accuracy,
            eval_time_s, index_vectors`.
        """
        configs: list[FusionFormulaConfig] = []
        if linear_alpha is not None:
            configs.append(self._default_config("linear", linear_alpha))
        configs.extend(self._default_config(formula) for formula in ("rumus1", "rumus2", "rumus3"))
        configs.extend(self._default_config("rumus4", alpha) for alpha in rumus4_alphas)

        rows: list[dict[str, object]] = []
        for config in configs:
            metrics, extras = self.evaluate(config, split=split)
            rows.append(self._row(config, metrics, extras))
            logger.info(
                "[%s alpha=%s] val F1-macro %.4f", config.formula, config.alpha, metrics["f1_macro"]
            )
        return pd.DataFrame(rows)

    def _default_config(
        self,
        formula: FusionFormula,
        alpha: float | None = None,
    ) -> FusionFormulaConfig:
        return FusionFormulaConfig(
            formula=formula, alpha=alpha, k=self.k, weighting=self.weighting
        )

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
