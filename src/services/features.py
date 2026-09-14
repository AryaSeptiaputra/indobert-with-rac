"""Ekstraksi dan caching fitur mean-pool dari encoder IndoBERT yang dibekukan.

RM-b dan RM-c sama-sama bekerja di atas embedding beku, dan encoder tidak pernah
berubah selama tuning. Karena itu ekstraksi cukup dijalankan sekali per encoder
lalu hasilnya dipakai ulang oleh puluhan konfigurasi head dan ratusan kombinasi
alpha/k. Cache dipisah per nama model agar dua encoder berbeda tidak saling
menimpa embedding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.config import SPLIT_NAMES, settings
from src.models.comment_dataset import GamblingCommentDataset
from src.models.heads import build_encoder, mean_pool
from src.services.evaluation import EfficiencyProfiler
from src.utils.io import read_json, write_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def model_slug(model_name: str) -> str:
    """Ubah nama repo Hub menjadi nama folder yang aman.

    Args:
        model_name: Nama repo, mis. "indobenchmark/indobert-base-p2".

    Returns:
        Nama folder tanpa pemisah path.
    """
    return model_name.replace("/", "__")


@dataclass
class FeatureSet:
    """Embedding beku dan label untuk ketiga split.

    Attributes:
        embeddings: Peta split ke array (N, H) float32.
        labels: Peta split ke array (N,) int64.
        hidden_dim: Dimensi embedding encoder.
        model_name: Encoder asal embedding ini.
        extract_time_s: Durasi ekstraksi; 0 bila dimuat dari cache.
        extract_peak_mem_mb: Memori GPU puncak saat ekstraksi.
        from_cache: True bila dimuat dari berkas, bukan dihitung ulang.
    """

    embeddings: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    hidden_dim: int
    model_name: str
    extract_time_s: float = 0.0
    extract_peak_mem_mb: float = 0.0
    from_cache: bool = False

    def __getitem__(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Pasangan (embedding, label) satu split.

        Args:
            split: Nama split.

        Returns:
            Tuple (embedding, label).

        Raises:
            KeyError: Kalau split tidak ada.
        """
        return self.embeddings[split], self.labels[split]


class FeatureExtractor:
    """Hasilkan embedding mean-pool dari encoder beku, dengan cache berbasis berkas.

    Args:
        tokenizer: Hasil `load_tokenizer()`.
        device: Device komputasi.
        model_name: Encoder dasar; `None` memakai `settings.base_model`.
        batch_size: Ukuran batch ekstraksi; `None` memakai konfigurasi.
        max_length: Panjang token maksimum; `None` memakai konfigurasi.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        model_name: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name or settings.base_model
        self.batch_size = batch_size or settings.feature_batch_size
        self.max_length = max_length or settings.max_length
        self.profiler = EfficiencyProfiler()

    def cache_dir(self, out_dir: Path) -> Path:
        """Folder cache embedding untuk encoder ini.

        Args:
            out_dir: Folder keluaran kampanye.

        Returns:
            Path folder cache khusus encoder ini.
        """
        return Path(out_dir) / "features" / model_slug(self.model_name)

    @torch.no_grad()
    def encode(
        self,
        encoder: PreTrainedModel,
        frame: pd.DataFrame,
        use_amp: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Hitung embedding mean-pool untuk seluruh baris satu DataFrame.

        Args:
            encoder: Encoder beku dalam mode eval.
            frame: Split yang akan diekstrak.
            use_amp: Aktifkan autocast; hanya berpengaruh di CUDA.

        Returns:
            Tuple (embedding (N, H) float32, label (N,) int64 atau `None` bila
            DataFrame tidak punya kolom label).

        Raises:
            ValueError: Kalau DataFrame kosong.
        """
        if frame.empty:
            raise ValueError("DataFrame kosong; tidak ada yang bisa diekstrak")

        dataset = GamblingCommentDataset.from_frame(
            frame, self.tokenizer, max_length=self.max_length
        )
        loader = DataLoader(dataset, batch_size=self.batch_size)
        amp_enabled = use_amp and self.device.type == "cuda"

        chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        has_labels = True

        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"].to(self.device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = encoder(**inputs)

            pooled = mean_pool(output.last_hidden_state, inputs["attention_mask"])
            chunks.append(pooled.float().cpu().numpy())

            if "labels" in batch:
                label_chunks.append(batch["labels"].numpy())
            else:
                has_labels = False

        embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
        labels = (
            np.concatenate(label_chunks, axis=0).astype(np.int64) if has_labels else None
        )
        return embeddings, labels

    def extract(self, frames: dict[str, pd.DataFrame]) -> FeatureSet:
        """Ekstrak embedding ketiga split dengan satu encoder yang dimuat sekali.

        Args:
            frames: Peta nama split ke DataFrame-nya.

        Returns:
            `FeatureSet` hasil ekstraksi.

        Raises:
            OSError: Kalau bobot encoder tidak bisa diunduh.
        """
        logger.info("Ekstraksi fitur beku dengan encoder %s", self.model_name)
        encoder = build_encoder(self.tokenizer, model_name=self.model_name).to(self.device)

        self.profiler.reset_peak_memory()
        started = time.perf_counter()

        embeddings: dict[str, np.ndarray] = {}
        labels: dict[str, np.ndarray] = {}
        hidden_dim = 0

        try:
            for split in SPLIT_NAMES:
                split_embeddings, split_labels = self.encode(encoder, frames[split])
                embeddings[split] = split_embeddings
                labels[split] = (
                    split_labels
                    if split_labels is not None
                    else np.zeros(len(split_embeddings), dtype=np.int64)
                )
                hidden_dim = split_embeddings.shape[1]
                logger.info("  %s: %s", split, split_embeddings.shape)
        finally:
            del encoder
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return FeatureSet(
            embeddings=embeddings,
            labels=labels,
            hidden_dim=hidden_dim,
            model_name=self.model_name,
            extract_time_s=round(time.perf_counter() - started, 2),
            extract_peak_mem_mb=round(self.profiler.peak_gpu_memory_mb(), 1),
            from_cache=False,
        )

    def load_or_extract(
        self,
        frames: dict[str, pd.DataFrame],
        out_dir: Path,
    ) -> FeatureSet:
        """Muat embedding dari cache, atau ekstrak lalu simpan bila belum ada.

        Args:
            frames: Peta nama split ke DataFrame-nya.
            out_dir: Folder keluaran kampanye.

        Returns:
            `FeatureSet`; `from_cache` menandai apakah hasilnya dari berkas.

        Raises:
            CorruptArtifactError: Kalau `extract_meta.json` ada tapi rusak.
            OSError: Kalau penulisan cache gagal.
        """
        cache = self.cache_dir(out_dir)
        cached = self._load_cache(cache)
        if cached is not None:
            logger.info("Fitur beku dimuat dari cache %s", cache)
            return cached

        feature_set = self.extract(frames)
        self._save_cache(cache, feature_set)
        return feature_set

    def _load_cache(self, cache: Path) -> FeatureSet | None:
        paths = {
            split: (cache / f"{split}_emb.npy", cache / f"{split}_label.npy")
            for split in SPLIT_NAMES
        }
        if not all(emb.exists() and lab.exists() for emb, lab in paths.values()):
            return None

        meta = read_json(cache / "extract_meta.json", default={}) or {}
        cached_name = meta.get("model_name")
        if cached_name is not None and cached_name != self.model_name:
            logger.warning(
                "Cache di %s berasal dari encoder %s, bukan %s; ekstraksi diulang",
                cache,
                cached_name,
                self.model_name,
            )
            return None

        try:
            embeddings = {s: np.load(paths[s][0]) for s in SPLIT_NAMES}
            labels = {s: np.load(paths[s][1]) for s in SPLIT_NAMES}
        except (OSError, ValueError):
            logger.warning(
                "Cache fitur di %s tidak bisa dibaca; ekstraksi diulang", cache, exc_info=True
            )
            return None

        return FeatureSet(
            embeddings=embeddings,
            labels=labels,
            hidden_dim=int(embeddings["train"].shape[1]),
            model_name=self.model_name,
            extract_time_s=float(meta.get("extract_time_s", 0.0)),
            extract_peak_mem_mb=float(meta.get("extract_peak_gpu_mem_mb", 0.0)),
            from_cache=True,
        )

    def _save_cache(self, cache: Path, feature_set: FeatureSet) -> None:
        cache.mkdir(parents=True, exist_ok=True)
        for split in SPLIT_NAMES:
            np.save(cache / f"{split}_emb.npy", feature_set.embeddings[split])
            np.save(cache / f"{split}_label.npy", feature_set.labels[split])

        write_json(
            cache / "extract_meta.json",
            {
                "model_name": feature_set.model_name,
                "hidden_dim": feature_set.hidden_dim,
                "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "extract_time_s": feature_set.extract_time_s,
                "extract_peak_gpu_mem_mb": feature_set.extract_peak_mem_mb,
            },
        )
        logger.info("Fitur beku disimpan ke %s", cache)
