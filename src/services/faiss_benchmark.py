"""Benchmark biaya membangun dan menelusuri indeks FAISS RM-c.

RM-c mengklaim nol waktu latih, tapi klaim itu hanya jujur bila biaya retrieval
ikut diukur. Modul ini memisahkan dua komponen biaya yang berbeda sifatnya:
pembangunan indeks (sekali, dari embedding train) dan penelusuran (setiap
inferensi, tumbuh mengikuti k).

Indeks bertipe flat/exact, jadi penelusuran bersifat brute force atas seluruh
6.588 vektor. Untuk ukuran itu biayanya sepele, dan hasilnya deterministik --
yang jauh lebih penting untuk penelitian daripada penghematan waktu dari indeks
aproksimasi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import psutil

from src.services.rac import l2_normalize
from src.utils.io import write_csv, write_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

BYTES_PER_MB = 1024**2
FLOAT32_BYTES = 4


@dataclass
class BenchmarkResult:
    """Hasil pengukuran pembangunan dan penelusuran indeks.

    Attributes:
        build: Metrik pembangunan indeks.
        search: Satu baris metrik penelusuran per nilai k.
    """

    build: dict[str, float | int]
    search: list[dict[str, float | int]] = field(default_factory=list)


class FaissBenchmark:
    """Ukur biaya indeks FAISS atas satu himpunan embedding train.

    Args:
        train_embeddings: Array (N, H) embedding split train.
        train_labels: Array (N,) label yang selaras dengan embedding.
        repeats: Jumlah pengulangan tiap pengukuran waktu.

    Raises:
        ValueError: Kalau jumlah embedding dan label tidak sama.
    """

    def __init__(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        repeats: int = 5,
    ) -> None:
        if len(train_embeddings) != len(train_labels):
            raise ValueError(
                f"jumlah embedding ({len(train_embeddings)}) != "
                f"jumlah label ({len(train_labels)})"
            )
        self.train_embeddings = train_embeddings
        self.train_labels = np.asarray(train_labels)
        self.repeats = repeats

    def measure_build(self) -> dict[str, float | int]:
        """Ukur waktu dan memori pembangunan indeks.

        Returns:
            Dict berisi jumlah vektor, dimensi, waktu bangun rata-rata,
            perkiraan ukuran indeks, dan kenaikan RSS proses.
        """
        normalized = l2_normalize(self.train_embeddings)
        n_vectors, dim = normalized.shape
        process = psutil.Process()

        rss_before = process.memory_info().rss
        durations = []
        index = None

        for _ in range(self.repeats):
            started = time.perf_counter()
            index = faiss.IndexFlatIP(dim)
            index.add(normalized)
            durations.append(time.perf_counter() - started)

        rss_after = process.memory_info().rss

        result = {
            "n_vectors": int(n_vectors),
            "dim": int(dim),
            "build_time_ms_mean": round(float(np.mean(durations)) * 1000, 4),
            "build_time_ms_min": round(float(np.min(durations)) * 1000, 4),
            # Indeks flat menyimpan seluruh vektor apa adanya sebagai float32.
            "index_size_mb": round(n_vectors * dim * FLOAT32_BYTES / BYTES_PER_MB, 3),
            "rss_delta_mb": round((rss_after - rss_before) / BYTES_PER_MB, 2),
            "index_ntotal": int(index.ntotal) if index is not None else 0,
        }
        logger.info(
            "Bangun indeks: %d vektor x %d dim dalam %.2f ms (%.2f MB)",
            result["n_vectors"], result["dim"],
            result["build_time_ms_mean"], result["index_size_mb"],
        )
        return result

    def measure_search(
        self,
        query_embeddings: np.ndarray,
        k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    ) -> list[dict[str, float | int]]:
        """Ukur waktu penelusuran untuk beberapa nilai k.

        Args:
            query_embeddings: Array (N, H) embedding query.
            k_values: Nilai k yang diuji.

        Returns:
            Satu dict per nilai k, memuat waktu total batch dan per-query.

        Raises:
            ValueError: Kalau ada k yang melebihi jumlah vektor train.
        """
        normalized_train = l2_normalize(self.train_embeddings)
        index = faiss.IndexFlatIP(normalized_train.shape[1])
        index.add(normalized_train)

        normalized_query = l2_normalize(query_embeddings)
        n_queries = len(normalized_query)

        rows: list[dict[str, float | int]] = []
        for k in k_values:
            if k > index.ntotal:
                raise ValueError(
                    f"k={k} melebihi jumlah vektor di indeks ({index.ntotal})"
                )

            durations = []
            for _ in range(self.repeats):
                started = time.perf_counter()
                index.search(normalized_query, k)
                durations.append(time.perf_counter() - started)

            mean_seconds = float(np.mean(durations))
            rows.append(
                {
                    "k": int(k),
                    "n_queries": int(n_queries),
                    "search_time_ms_batch": round(mean_seconds * 1000, 4),
                    "search_time_us_per_query": round(
                        mean_seconds / n_queries * 1_000_000, 4
                    ),
                }
            )
            logger.info(
                "Telusur k=%d: %.3f ms untuk %d query (%.2f us/query)",
                k, rows[-1]["search_time_ms_batch"], n_queries,
                rows[-1]["search_time_us_per_query"],
            )

        return rows

    def run(
        self,
        query_embeddings: np.ndarray,
        k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
        out_dir: Path | None = None,
    ) -> BenchmarkResult:
        """Jalankan pengukuran pembangunan dan penelusuran, lalu simpan hasilnya.

        Args:
            query_embeddings: Array embedding query.
            k_values: Nilai k yang diuji.
            out_dir: Folder keluaran; `None` berarti hasil tidak ditulis ke disk.

        Returns:
            `BenchmarkResult` berisi kedua bagian pengukuran.
        """
        result = BenchmarkResult(
            build=self.measure_build(),
            search=self.measure_search(query_embeddings, k_values),
        )

        if out_dir is not None:
            out_dir = Path(out_dir)
            write_json(out_dir / "faiss_build.json", result.build)
            write_csv(out_dir / "faiss_search.csv", pd.DataFrame(result.search))
            logger.info("Hasil benchmark FAISS ditulis ke %s", out_dir)

        return result
