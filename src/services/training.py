"""Engine training dan evaluasi satu-konfigurasi untuk RM-a, RM-b, dan RM-c.

Alur penelitian ini human-in-the-loop, bukan pencarian otomatis:

    tentukan nilai -> jalankan satu run -> analisis -> ubah nilai -> ulang

Karena itu modul ini sengaja TIDAK memuat algoritma pencarian. Satu panggilan
sama dengan satu konfigurasi dan satu baris riwayat run, dengan alasan pemilihan
nilai dicatat manual di kolom `catatan`.

Seleksi hyperparameter memakai split validation. Split test hanya disentuh bila
diminta eksplisit, agar tidak ada kebocoran pemilihan model lewat test.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_utils import PreTrainedModel

from src.config import settings
from src.models.comment_dataset import GamblingCommentDataset
from src.models.heads import build_finetune_model, build_head
from src.models.schemas import RMAConfig, RMBConfig, RMCConfig
from src.services.data import ExperimentData
from src.services.evaluation import ClassificationEvaluator, EfficiencyProfiler
from src.services.features import FeatureSet
from src.services.rac import RACClassifier, softmax
from src.utils.logger import setup_logger
from src.utils.seeding import set_seed

logger = setup_logger(__name__)

# Presisi pembulatan kolom hasil; cukup untuk membedakan run tapi tidak
# menyimpan derau floating point ke CSV.
LOSS_PRECISION = 6
TIME_PRECISION = 2
MEMORY_PRECISION = 1


@dataclass
class TrainingResult:
    """Hasil satu run training: metrik terbaik, kurva per-epoch, dan biaya komputasi.

    Attributes:
        best_metrics: Metrik validation pada epoch terbaik.
        history: Satu dict per epoch, untuk mendeteksi overfitting.
        best_epoch: Epoch dengan F1-macro validation tertinggi.
        train_time_s: Total durasi training.
        peak_mem_mb: Memori GPU puncak selama training.
        trainable_params: Jumlah parameter yang benar-benar diperbarui.
        best_state: Salinan CPU state_dict pada epoch terbaik.
        extras: Kolom tambahan spesifik skenario.
    """

    best_metrics: dict[str, float | int]
    history: list[dict[str, float | int]]
    best_epoch: int
    train_time_s: float
    peak_mem_mb: float
    trainable_params: int
    best_state: dict[str, torch.Tensor] | None = None
    extras: dict[str, object] = field(default_factory=dict)


def history_row(
    epoch: int,
    train_loss: float,
    metrics: dict[str, float | int],
) -> dict[str, float | int]:
    """Susun satu baris kurva per-epoch.

    Args:
        epoch: Nomor epoch, mulai dari 1.
        train_loss: Rata-rata loss training pada epoch itu.
        metrics: Metrik validation epoch itu.

    Returns:
        Dict satu baris untuk berkas history.
    """
    return {
        "epoch": epoch,
        "train_loss": round(train_loss, LOSS_PRECISION),
        "val_f1_macro": metrics["f1_macro"],
        "val_acc": metrics["accuracy"],
        "val_f1_judi": metrics["f1_class1"],
        "val_precision_judi": metrics["precision_class1"],
        "val_recall_judi": metrics["recall_class1"],
    }


class RMATrainer:
    """RM-a: full fine-tuning seluruh parameter IndoBERT.

    Batch efektif dicapai lewat akumulasi gradien di atas micro-batch, sehingga
    konfigurasi `batch=32` tetap bisa dijalankan pada GPU 4 GB tanpa mengubah
    arti hyperparameter-nya.

    Args:
        data: Split, tokenizer, class weight, dan device.
    """

    def __init__(self, data: ExperimentData) -> None:
        self.data = data
        self.evaluator = ClassificationEvaluator()
        self.profiler = EfficiencyProfiler()

    def loader(self, frame: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
        """Bungkus satu split menjadi `DataLoader`.

        Args:
            frame: DataFrame split.
            batch_size: Ukuran micro-batch.
            shuffle: Acak urutan; True hanya untuk train.

        Returns:
            DataLoader siap iterasi.
        """
        dataset = GamblingCommentDataset.from_frame(
            frame, self.data.tokenizer, max_length=self.data.max_length
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.data.num_workers,
            pin_memory=True,
        )

    @torch.no_grad()
    def predict(
        self,
        model: PreTrainedModel,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Jalankan inferensi atas seluruh loader.

        Args:
            model: Model terlatih.
            loader: DataLoader split yang dievaluasi.

        Returns:
            Tuple (label sebenarnya, prediksi, probabilitas kelas judi). Kolom
            ketiga dipakai untuk PR curve tanpa perlu forward pass tambahan.
        """
        model.eval()
        device = self.data.device
        use_amp = device.type == "cuda"

        truths, predictions, positives = [], [], []
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            token_type = batch.get("token_type_ids")
            inputs["token_type_ids"] = (
                token_type.to(device) if token_type is not None else None
            )

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(**inputs).logits

            positives.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            predictions.append(logits.argmax(1).cpu().numpy())
            truths.append(batch["labels"].numpy())

        return (
            np.concatenate(truths),
            np.concatenate(predictions),
            np.concatenate(positives),
        )

    def train(
        self,
        config: RMAConfig,
        keep_best: bool = True,
    ) -> tuple[TrainingResult, PreTrainedModel]:
        """Latih RM-a untuk satu konfigurasi.

        Args:
            config: Hyperparameter yang tervalidasi.
            keep_best: Simpan salinan bobot pada epoch terbaik.

        Returns:
            Tuple (hasil training, model pada kondisi akhir training).

        Raises:
            torch.cuda.OutOfMemoryError: Kalau micro-batch masih terlalu besar
                untuk VRAM yang tersedia.
        """
        set_seed(config.seed)
        device = self.data.device

        model = build_finetune_model(
            self.data.tokenizer, model_name=self.data.model_name
        ).to(device)

        micro_batch = config.effective_micro_batch
        accum = config.grad_accum
        train_loader = self.loader(self.data.train, micro_batch, shuffle=True)
        val_loader = self.loader(self.data.val, micro_batch, shuffle=False)

        criterion = nn.CrossEntropyLoss(weight=self.data.class_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        total_steps = math.ceil(len(train_loader) / accum) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(config.warmup_ratio * total_steps), total_steps
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        self.profiler.reset_peak_memory()
        started = time.perf_counter()

        best_f1 = -1.0
        best_metrics: dict[str, float | int] = {}
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, float | int]] = []

        for epoch in range(1, config.epochs + 1):
            running_loss = self._train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler, accum
            )
            truths, predictions, _ = self.predict(model, val_loader)
            metrics = self.evaluator.metrics(truths, predictions)

            history.append(
                history_row(epoch, running_loss / len(self.data.train), metrics)
            )
            logger.info(
                "[RM-a] epoch %d/%d val F1-macro %.4f",
                epoch,
                config.epochs,
                metrics["f1_macro"],
            )

            if metrics["f1_macro"] > best_f1:
                best_f1 = float(metrics["f1_macro"])
                best_metrics, best_epoch = metrics, epoch
                if keep_best:
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

        result = TrainingResult(
            best_metrics=best_metrics,
            history=history,
            best_epoch=best_epoch,
            train_time_s=round(time.perf_counter() - started, TIME_PRECISION),
            peak_mem_mb=round(self.profiler.peak_gpu_memory_mb(), MEMORY_PRECISION),
            trainable_params=int(
                self.profiler.count_parameters(model)["trainable_params"]
            ),
            best_state=best_state,
            extras={
                "micro_batch_eff": micro_batch,
                "grad_accum": accum,
            },
        )
        return result, model

    def evaluate_test(
        self,
        model: PreTrainedModel,
        batch_size: int | None = None,
    ) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
        """Evaluasi model pada split test.

        Args:
            model: Model yang sudah dimuati bobot terbaik.
            batch_size: Ukuran batch inferensi; `None` memakai
                `settings.feature_batch_size`.

        Returns:
            Tuple (metrik, label sebenarnya, prediksi, probabilitas kelas judi).
        """
        loader = self.loader(
            self.data.test, batch_size or settings.feature_batch_size, shuffle=False
        )
        truths, predictions, positives = self.predict(model, loader)
        return self.evaluator.metrics(truths, predictions), truths, predictions, positives

    def _train_one_epoch(
        self,
        model: PreTrainedModel,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        scaler: torch.amp.GradScaler,
        accum: int,
    ) -> float:
        device = self.data.device
        use_amp = device.type == "cuda"
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        n_batches = len(loader)

        for index, batch in enumerate(loader):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            token_type = batch.get("token_type_ids")
            inputs["token_type_ids"] = (
                token_type.to(device) if token_type is not None else None
            )
            targets = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(**inputs).logits
                # Dibagi accum agar gradien terakumulasi setara satu batch besar.
                loss = criterion(logits, targets) / accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * accum * targets.size(0)

            if (index + 1) % accum == 0 or (index + 1) == n_batches:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        return running_loss


class RMBTrainer:
    """RM-b: latih classification head di atas embedding beku.

    Encoder tidak pernah disentuh, sehingga satu epoch hanya berupa perkalian
    matriks kecil atas fitur yang sudah dihitung. Inilah sumber utama keunggulan
    efisiensi RM-b terhadap RM-a.

    Args:
        features: Embedding beku ketiga split.
        class_weights: Tensor bobot kelas di device yang sama.
        device: Device komputasi.
    """

    def __init__(
        self,
        features: FeatureSet,
        class_weights: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.features = features
        self.class_weights = class_weights
        self.device = device
        self.evaluator = ClassificationEvaluator()
        self.profiler = EfficiencyProfiler()

    def train(self, config: RMBConfig, eval_test: bool = False) -> TrainingResult:
        """Latih head untuk satu konfigurasi.

        Args:
            config: Hyperparameter yang tervalidasi.
            eval_test: Bila True, evaluasi juga pada split test memakai bobot
                epoch terbaik.

        Returns:
            Hasil training; `extras` memuat `hidden_size`, `infer_latency_ms`,
            dan (bila diminta) metrik serta prediksi test.
        """
        set_seed(config.seed)

        train_emb, train_lab = self.features["train"]
        val_emb, val_lab = self.features["val"]
        hidden_size = int(train_emb.shape[1])

        head = build_head(
            config.head_arch,
            hidden_size=hidden_size,
            dropout=config.dropout,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        x_train = torch.tensor(train_emb, device=self.device)
        y_train = torch.tensor(train_lab, device=self.device)
        x_val = torch.tensor(val_emb, device=self.device)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=config.batch, shuffle=True
        )

        self.profiler.reset_peak_memory()
        started = time.perf_counter()

        best_f1 = -1.0
        best_metrics: dict[str, float | int] = {}
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, float | int]] = []

        for epoch in range(1, config.epochs + 1):
            head.train()
            running_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(head(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)

            head.eval()
            with torch.no_grad():
                predictions = head(x_val).argmax(1).cpu().numpy()
            metrics = self.evaluator.metrics(val_lab, predictions)

            history.append(
                history_row(epoch, running_loss / len(x_train), metrics)
            )

            if metrics["f1_macro"] > best_f1:
                best_f1 = float(metrics["f1_macro"])
                best_metrics, best_epoch = metrics, epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                }

        train_time_s = round(time.perf_counter() - started, TIME_PRECISION)
        peak_mem_mb = round(self.profiler.peak_gpu_memory_mb(), MEMORY_PRECISION)

        head.eval()
        latency_ms = self.profiler.measure_latency(lambda: head(x_val[:1]))

        extras: dict[str, object] = {
            "hidden_size": hidden_size,
            "infer_latency_ms": round(latency_ms, 4),
        }

        if eval_test and best_state is not None:
            extras.update(self._evaluate_test(head, best_state))

        return TrainingResult(
            best_metrics=best_metrics,
            history=history,
            best_epoch=best_epoch,
            train_time_s=train_time_s,
            peak_mem_mb=peak_mem_mb,
            trainable_params=int(self.profiler.count_parameters(head)["trainable_params"]),
            best_state=best_state,
            extras=extras,
        )

    def _evaluate_test(
        self,
        head: nn.Module,
        best_state: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        head.load_state_dict(best_state)
        head.to(self.device).eval()
        test_emb, test_lab = self.features["test"]

        with torch.no_grad():
            logits = head(torch.tensor(test_emb, device=self.device))
            predictions = logits.argmax(1).cpu().numpy()
            positives = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()

        return {
            "test_metrics": self.evaluator.metrics(test_lab, predictions),
            "test_pred": predictions,
            "test_pred_proba": positives,
        }


class RMCEvaluator:
    """RM-c: evaluasi RAC di atas head RM-b yang sudah terlatih, tanpa training.

    Args:
        features: Embedding beku ketiga split.
        head: Head RM-b terbaik yang sudah dimuati bobot.
        device: Device komputasi.
    """

    def __init__(
        self,
        features: FeatureSet,
        head: nn.Module,
        device: torch.device,
    ) -> None:
        self.features = features
        self.head = head
        self.device = device
        self.evaluator = ClassificationEvaluator()

    def evaluate(
        self,
        config: RMCConfig,
        split: str = "val",
    ) -> tuple[dict[str, float | int], dict[str, object]]:
        """Evaluasi satu konfigurasi (alpha, k, weighting) pada satu split.

        Args:
            config: Hyperparameter RAC yang tervalidasi.
            split: Split yang dievaluasi, biasanya "val" saat tuning.

        Returns:
            Tuple (metrik, extras). `extras` memuat `eval_time_s`,
            `index_vectors`, `preds`, dan `p_judi`.

        Raises:
            ValueError: Kalau `k` melebihi jumlah vektor train di indeks.
        """
        query_emb, query_lab = self.features[split]
        train_emb, train_lab = self.features["train"]

        self.head.eval().to(self.device)
        with torch.no_grad():
            logits = self.head(torch.tensor(query_emb, device=self.device))
        p_bert = softmax(logits.cpu().numpy())

        classifier = RACClassifier(
            alpha=config.alpha, k=config.k, weighting=config.weighting
        ).fit(train_emb, train_lab)

        started = time.perf_counter()
        predictions, p_final = classifier.predict(query_emb, p_bert)
        elapsed = round(time.perf_counter() - started, TIME_PRECISION)

        extras: dict[str, object] = {
            "eval_time_s": elapsed,
            "index_vectors": classifier.index_size,
            "index_type": classifier.index_type,
            "preds": predictions,
            "p_judi": p_final[:, 1],
        }
        return self.evaluator.metrics(query_lab, predictions), extras
