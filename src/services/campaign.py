"""Orkestrasi kampanye tuning: menjalankan run, mencatat hasil, dan benchmark final.

Satu pemanggilan `run` sama dengan satu konfigurasi dan satu baris riwayat.
Tidak ada algoritma pencarian di sini: urutan konfigurasi ditentukan manusia,
dituliskan di `tuning_grids/`, lalu diserahkan ke `run_batch`.

Kegagalan satu konfigurasi di dalam batch diisolasi ke `runs_{scenario}_errors.csv`
supaya sisa antrean tetap jalan; kampanye lima jam tidak boleh batal karena satu
nilai yang keliru di baris ke-tiga puluh.

Seluruh seleksi hyperparameter memakai split validation. Split test hanya dibuka
di `run_final`, sekali, untuk ketiga skenario dalam satu sesi GPU yang sama --
syarat agar angka efisiensi di Bab 4 sah dibandingkan.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from src.config import SCENARIOS, settings
from src.models.comment_dataset import GamblingCommentDataset
from src.models.heads import build_encoder, build_finetune_model, build_head, mean_pool
from src.models.schemas import CONFIG_MODELS, RunRequest, parse_config
from src.services.data import ExperimentData
from src.services.evaluation import RUN_METRIC_KEYS, ClassificationEvaluator, EfficiencyProfiler
from src.services.features import FeatureExtractor, FeatureSet
from src.services.rac import RACClassifier, softmax
from src.services.reporting import FigureReporter
from src.services.run_log import BestTracker, HistoryWriter, RunLogger
from src.services.training import RMATrainer, RMBTrainer, RMCEvaluator
from src.utils.io import read_csv, write_csv, write_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

METRIC_PRECISION = 6

# Ambang kriteria sukses: strategi ringan dianggap kompetitif bila memenuhi
# minimal dua dari tiga syarat ini.
MAX_F1_GAP_PP = 3.0
MIN_PARAM_REDUCTION_PCT = 90.0
MIN_TIME_REDUCTION_PCT = 50.0
MIN_CRITERIA_PASSED = 2

DISPLAY_NAMES = {"rma": "RM-a", "rmb": "RM-b", "rmc": "RM-c"}

# Presisi pembulatan saat membandingkan nilai float antar konfigurasi. Cukup
# ketat untuk membedakan 1e-5 dari 2e-5, cukup longgar untuk menyerap galat
# pembacaan ulang dari CSV.
SIGNATURE_PRECISION = 12


class CampaignRunner:
    """Jalankan dan catat run tuning untuk satu folder keluaran.

    Data, tokenizer, dan fitur beku dimuat malas lalu dipakai ulang, sehingga
    menjalankan 67 konfigurasi RM-c hanya memuat encoder sekali.

    Args:
        out_dir: Folder keluaran; `None` memakai `settings.default_out_dir`.
        smoke: Jalankan di atas subset kecil untuk membuktikan pipeline berjalan.
        model_name: Encoder dasar; `None` memakai `settings.base_model`.
        device: Device komputasi; `None` memilih otomatis.
    """

    def __init__(
        self,
        out_dir: str | Path | None = None,
        smoke: bool = False,
        model_name: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.out_dir = settings.resolve_out_dir(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.smoke = smoke
        self.model_name = model_name or settings.base_model
        self._device_request = device

        self.reporter = FigureReporter(self.out_dir)
        self.history = HistoryWriter(self.out_dir)
        self.best = BestTracker(self.out_dir)
        self.evaluator = ClassificationEvaluator()
        self.profiler = EfficiencyProfiler()

        self._data: ExperimentData | None = None
        self._features: FeatureSet | None = None

    @property
    def data(self) -> ExperimentData:
        """Split, tokenizer, dan class weight; dimuat sekali saat pertama diakses."""
        if self._data is None:
            self._data = ExperimentData.load(
                model_name=self.model_name, device=self._device_request, smoke=self.smoke
            )
        return self._data

    @property
    def device(self) -> torch.device:
        """Device komputasi yang dipakai kampanye ini."""
        return self.data.device

    @property
    def features(self) -> FeatureSet:
        """Embedding beku ketiga split, diekstrak sekali lalu di-cache ke disk."""
        if self._features is None:
            extractor = FeatureExtractor(
                self.data.tokenizer, self.device, model_name=self.model_name
            )
            self._features = extractor.load_or_extract(self.data.frames, self.out_dir)
        return self._features

    @property
    def checkpoint_dir(self) -> Path:
        """Folder checkpoint, dibuat bila belum ada."""
        path = self.out_dir / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(
        self,
        scenario: str,
        config: dict[str, object] | None = None,
        note: str = "",
        eval_test: bool = False,
        batch_id: str = "",
    ) -> dict[str, object]:
        """Jalankan satu konfigurasi dan catat satu baris riwayat.

        Args:
            scenario: "rma", "rmb", atau "rmc".
            config: Hyperparameter; nilai yang tidak diisi memakai default skenario.
            note: Alasan konfigurasi ini dicoba, disimpan ke kolom `catatan`.
            eval_test: Buka split test untuk run ini. Biarkan False selama tuning.
            batch_id: Penanda batch asal, kosong untuk run tunggal.

        Returns:
            Baris riwayat yang sudah tercatat, termasuk `run_id` dan kolom turunan.

        Raises:
            ValueError: Kalau skenario tidak dikenal atau nilai hyperparameter
                di luar batas.
            FileNotFoundError: Kalau RM-c dijalankan sebelum RM-b punya checkpoint.
        """
        if scenario not in SCENARIOS:
            raise ValueError(f"skenario tak dikenal: {scenario!r} (harus salah satu {SCENARIOS})")

        parsed = parse_config(scenario, config)
        run_logger = RunLogger(self.out_dir / f"runs_{scenario}.csv")
        run_id = run_logger.next_id()
        logger.info("[%s] RUN #%d %s", scenario, run_id, parsed.model_dump())

        handler = {
            "rma": self._run_rma,
            "rmb": self._run_rmb,
            "rmc": self._run_rmc,
        }[scenario]
        return handler(parsed, run_id, run_logger, note, eval_test, batch_id)

    def config_signature(
        self,
        scenario: str,
        config: dict[str, object] | None,
    ) -> tuple[tuple[str, object], ...]:
        """Tanda pengenal satu konfigurasi, dipakai untuk mendeteksi run kembar.

        Konfigurasi dilengkapi nilai default lebih dulu, sehingga `{"lr": 2e-5}`
        dan konfigurasi lengkap dengan nilai yang sama menghasilkan tanda yang
        identik.

        Untuk RM-a, `micro_batch` dinormalkan ke nilai EFEKTIFNYA
        (`min(batch, micro_batch)`). Nilai efektif itulah yang menentukan ukuran
        batch di `DataLoader`, dan karena itu menentukan lintasan training;
        `micro_batch=32` pada `batch=16` menghasilkan run yang persis sama
        dengan `micro_batch=16`.

        Args:
            scenario: Kode skenario.
            config: Hyperparameter mentah; `None` berarti seluruh default.

        Returns:
            Tuple pasangan (nama, nilai) yang terurut dan bisa di-hash.

        Raises:
            ValueError: Kalau skenario tidak dikenal.
            pydantic.ValidationError: Kalau ada nilai di luar batas.
        """
        parsed = parse_config(scenario, config)
        values = parsed.model_dump()
        if scenario == "rma":
            values["micro_batch"] = parsed.effective_micro_batch
        return self._normalise(values)

    def completed_signatures(self, scenario: str) -> set[tuple[tuple[str, object], ...]]:
        """Tanda pengenal seluruh konfigurasi yang sudah tercatat di riwayat.

        Args:
            scenario: Kode skenario.

        Returns:
            Himpunan tanda pengenal; kosong bila riwayat belum ada.

        Raises:
            CorruptArtifactError: Kalau berkas riwayat ada tapi tidak bisa diurai.
        """
        frame = read_csv(self.out_dir / f"runs_{scenario}.csv")
        if frame.empty:
            return set()

        fields = list(CONFIG_MODELS[scenario].model_fields)
        available = [name for name in fields if name in frame.columns]
        if not available:
            return set()

        signatures = set()
        for row in frame[available + self._effective_columns(scenario, frame)].to_dict(
            "records"
        ):
            values = {name: row[name] for name in available}
            if scenario == "rma" and "micro_batch_eff" in row:
                values["micro_batch"] = row["micro_batch_eff"]
            signatures.add(self._normalise(values))
        return signatures

    def pending_requests(
        self,
        scenario: str,
        requests: list[RunRequest | dict[str, object]],
    ) -> list[RunRequest]:
        """Saring konfigurasi yang BELUM pernah dijalankan.

        Args:
            scenario: Kode skenario.
            requests: Daftar permintaan run.

        Returns:
            Sublist berisi permintaan yang belum ada padanannya di riwayat,
            urutannya dipertahankan. Konfigurasi yang nilainya tidak sah ikut
            diteruskan, bukan dilempar dari sini: kegagalannya harus dicatat
            oleh `run_batch` ke `runs_{scenario}_errors.csv` bersama konteks
            batch-nya, bukan membatalkan seluruh antrean.
        """
        done = self.completed_signatures(scenario)
        pending: list[RunRequest] = []
        seen: set[tuple[tuple[str, object], ...]] = set()

        for item in self._as_requests(requests):
            try:
                signature = self.config_signature(scenario, item.config)
            except (ValueError, ValidationError):
                pending.append(item)
                continue
            if signature in done or signature in seen:
                continue
            seen.add(signature)
            pending.append(item)
        return pending

    def run_batch(
        self,
        scenario: str,
        requests: list[RunRequest | dict[str, object]],
        batch_id: str = "",
        resume: bool = True,
    ) -> pd.DataFrame:
        """Jalankan sederet konfigurasi berurutan dengan isolasi kegagalan.

        Args:
            scenario: Kode skenario.
            requests: Daftar `RunRequest`, atau dict berisi kunci `config`,
                `note`, dan `eval_test`.
            batch_id: Penanda batch; kosong berarti dibuat dari waktu sekarang.
            resume: Lewati konfigurasi yang sudah ada di riwayat. Aktif secara
                default supaya batch yang terputus bisa dijalankan ulang apa
                adanya tanpa mengulang pekerjaan yang sudah selesai. Setel False
                bila memang ingin mengukur ulang konfigurasi yang sama.

        Returns:
            DataFrame berisi satu baris per konfigurasi yang berhasil dijalankan
            pada pemanggilan ini; kosong bila semuanya sudah pernah dijalankan.

        Raises:
            ValueError: Kalau skenario tidak dikenal.
        """
        if scenario not in SCENARIOS:
            raise ValueError(f"skenario tak dikenal: {scenario!r}")

        batch_id = batch_id or f"{scenario}_batch_{time.strftime('%Y%m%d_%H%M%S')}"
        requested = self._as_requests(requests)

        if resume:
            items = self.pending_requests(scenario, requested)
            skipped = len(requested) - len(items)
            if skipped:
                logger.info(
                    "Batch %s: %d dari %d konfigurasi sudah ada di riwayat, dilewati",
                    batch_id, skipped, len(requested),
                )
            if not items:
                logger.info("Batch %s: tidak ada konfigurasi baru", batch_id)
                return pd.DataFrame()
        else:
            items = requested

        logger.info("Batch %s: %d konfigurasi akan dijalankan", batch_id, len(items))

        rows: list[dict[str, object]] = []
        started = time.perf_counter()

        for sequence, item in enumerate(items, start=1):
            elapsed = time.perf_counter() - started
            eta = (elapsed / max(sequence - 1, 1)) * (len(items) - sequence + 1)
            logger.info(
                "Batch %s: %d/%d (perkiraan sisa %.1f menit)",
                batch_id, sequence, len(items), eta / 60,
            )
            try:
                rows.append(
                    self.run(
                        scenario,
                        config=item.config,
                        note=item.note,
                        eval_test=item.eval_test,
                        batch_id=batch_id,
                    )
                )
            except (ValueError, RuntimeError, OSError, torch.cuda.OutOfMemoryError) as exc:
                logger.error(
                    "Konfigurasi %d/%d gagal, batch dilanjutkan", sequence, len(items),
                    exc_info=True,
                )
                RunLogger(self.out_dir / f"runs_{scenario}.csv").log_error(
                    scenario, batch_id, sequence, item.config, item.note, exc
                )

        self.reporter.refresh_scenario(scenario)
        self.reporter.write_summary()
        logger.info(
            "Batch %s selesai: %d/%d berhasil dalam %.1f menit",
            batch_id, len(rows), len(items), (time.perf_counter() - started) / 60,
        )
        return pd.DataFrame(rows)

    def run_final(self) -> dict[str, pd.DataFrame]:
        """Evaluasi test dan benchmark inferensi ketiga skenario dalam satu sesi.

        Memakai konfigurasi terbaik tiap skenario menurut `best.json`. Ketiga
        model diukur berurutan pada GPU yang sama tanpa proses lain di antaranya,
        karena angka waktu dan memori baru sebanding pada kondisi itu.

        Returns:
            Dict berisi DataFrame `comparison`, `benchmark`, dan `criteria`.

        Raises:
            RuntimeError: Kalau ada skenario yang belum punya run.
            FileNotFoundError: Kalau checkpoint yang dibutuhkan tidak ada.
        """
        missing = [s for s in SCENARIOS if not self.best.get(s)]
        if missing:
            raise RuntimeError(
                f"skenario {missing} belum punya run; jalankan tuning dulu sebelum Final"
            )

        logger.info("Benchmark final dimulai (satu sesi, device %s)", self.device)
        model = self._load_rma_model()
        head, head_config = self._load_best_head()
        encoder = build_encoder(self.data.tokenizer, model_name=self.model_name).to(self.device)
        rmc_config = parse_config("rmc", self._load_checkpoint("rmc_best.pt")["config"])

        test_metrics = self._final_test_metrics(model, head, rmc_config)
        benchmark = self._inference_benchmark(model, encoder, head, rmc_config)
        comparison = self._comparison_table(test_metrics, benchmark)
        criteria = self._success_criteria(test_metrics)

        self.reporter.final_inference_bar_chart()
        self.reporter.write_summary()
        return {"comparison": comparison, "benchmark": benchmark, "criteria": criteria}

    def write_hardware(self) -> dict[str, object]:
        """Catat spesifikasi hardware dan versi pustaka ke `hardware.json`.

        Angka efisiensi hanya bisa ditafsirkan bersama konteks ini, jadi berkasnya
        ditulis di folder keluaran yang sama dengan hasilnya.

        Returns:
            Dict informasi hardware yang ditulis.
        """
        import transformers

        info: dict[str, object] = {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vram_total_mb": (
                round(torch.cuda.get_device_properties(0).total_memory / 1024**2)
                if torch.cuda.is_available()
                else 0
            ),
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            info["nvidia_smi"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            logger.warning("nvidia-smi tidak tersedia; bagian itu dilewati")

        write_json(self.out_dir / "hardware.json", info)
        return info

    # ------------------------------------------------------------------
    # Handler per skenario
    # ------------------------------------------------------------------

    def _run_rma(
        self,
        config,
        run_id: int,
        run_logger: RunLogger,
        note: str,
        eval_test: bool,
        batch_id: str,
    ) -> dict[str, object]:
        result, model = RMATrainer(self.data).train(config)

        row = {
            "scenario": "rma",
            "batch_id": batch_id,
            "model_name": self.model_name,
            **config.model_dump(),
            "micro_batch_eff": result.extras["micro_batch_eff"],
            "grad_accum": result.extras["grad_accum"],
            **self._metric_columns(result.best_metrics),
            "best_epoch": result.best_epoch,
            "train_time_s": result.train_time_s,
            "peak_mem_mb": result.peak_mem_mb,
            "trainable_params": result.trainable_params,
            "catatan": note,
        }

        if eval_test and result.best_state is not None:
            model.load_state_dict(result.best_state)
            trainer = RMATrainer(self.data)
            metrics, truths, predictions, positives = trainer.evaluate_test(model)
            row.update(self._metric_columns(metrics, prefix="test"))
            self._plot_run_test("rma", run_id, truths, predictions, positives)

        row = run_logger.log(row)
        self.history.append("rma", run_id, result.history)
        self.reporter.training_curve("rma", run_id, result.history)

        promoted = self.best.update(
            "rma",
            {
                "run_id": run_id,
                "config": config.model_dump(),
                "val_f1_macro": result.best_metrics["f1_macro"],
                "train_time_s": result.train_time_s,
                "trainable_params": result.trainable_params,
                "peak_mem_mb": result.peak_mem_mb,
                "model_name": self.model_name,
            },
        )
        if promoted and result.best_state is not None:
            torch.save(
                {
                    "model_state": result.best_state,
                    "config": config.model_dump(),
                    "run_id": run_id,
                    "val_f1_macro": result.best_metrics["f1_macro"],
                    "model_name": self.model_name,
                },
                self.checkpoint_dir / "rma_best.pt",
            )
            model.load_state_dict(result.best_state)
            trainer = RMATrainer(self.data)
            truths, predictions, positives = trainer.predict(
                model, trainer.loader(self.data.val, config.effective_micro_batch, False)
            )
            self._plot_champion("rma", run_id, truths, predictions, positives)

        logger.info(
            "[rma] RUN #%d val F1-macro %.4f (epoch terbaik %d)",
            run_id, result.best_metrics["f1_macro"], result.best_epoch,
        )
        return row

    def _run_rmb(
        self,
        config,
        run_id: int,
        run_logger: RunLogger,
        note: str,
        eval_test: bool,
        batch_id: str,
    ) -> dict[str, object]:
        features = self.features
        result = RMBTrainer(features, self.data.class_weights, self.device).train(
            config, eval_test=eval_test
        )

        row = {
            "scenario": "rmb",
            "batch_id": batch_id,
            "model_name": self.model_name,
            **config.model_dump(),
            **self._metric_columns(result.best_metrics),
            "best_epoch": result.best_epoch,
            "head_train_time_s": result.train_time_s,
            "extract_time_s": features.extract_time_s,
            # Biaya latih RM-b yang jujur mencakup ekstraksi fitur satu kali,
            # walau biaya itu diamortisasi ke seluruh konfigurasi head.
            "train_time_s": round(result.train_time_s + features.extract_time_s, 2),
            "trainable_params": result.trainable_params,
            "peak_mem_mb": result.peak_mem_mb,
            "infer_latency_ms": result.extras["infer_latency_ms"],
            "catatan": note,
        }

        if eval_test and "test_metrics" in result.extras:
            row.update(self._metric_columns(result.extras["test_metrics"], prefix="test"))
            self._plot_run_test(
                "rmb",
                run_id,
                features.labels["test"],
                result.extras["test_pred"],
                result.extras["test_pred_proba"],
            )

        row = run_logger.log(row)
        self.history.append("rmb", run_id, result.history)
        self.reporter.training_curve("rmb", run_id, result.history)

        promoted = self.best.update(
            "rmb",
            {
                "run_id": run_id,
                "config": config.model_dump(),
                "val_f1_macro": result.best_metrics["f1_macro"],
                "train_time_s": row["train_time_s"],
                "trainable_params": result.trainable_params,
                "peak_mem_mb": result.peak_mem_mb,
                "infer_latency_ms": result.extras["infer_latency_ms"],
                "model_name": self.model_name,
            },
        )
        if promoted and result.best_state is not None:
            torch.save(
                {
                    "head_state": result.best_state,
                    "config": config.model_dump(),
                    "run_id": run_id,
                    "hidden_size": result.extras["hidden_size"],
                    "val_f1_macro": result.best_metrics["f1_macro"],
                    "model_name": self.model_name,
                },
                self.checkpoint_dir / "rmb_best.pt",
            )
            self._plot_rmb_champion(config, result, run_id)

        logger.info(
            "[rmb] RUN #%d val F1-macro %.4f (epoch terbaik %d)",
            run_id, result.best_metrics["f1_macro"], result.best_epoch,
        )
        return row

    def _run_rmc(
        self,
        config,
        run_id: int,
        run_logger: RunLogger,
        note: str,
        eval_test: bool,
        batch_id: str,
    ) -> dict[str, object]:
        features = self.features
        head, head_config = self._load_best_head()
        evaluator = RMCEvaluator(features, head, self.device)

        metrics, extras = evaluator.evaluate(config, split="val")

        row = {
            "scenario": "rmc",
            "batch_id": batch_id,
            "model_name": self.model_name,
            **config.model_dump(),
            "head_arch_rmb": head_config.get("head_arch"),
            **self._metric_columns(metrics),
            "index_vectors": extras["index_vectors"],
            "eval_time_s": extras["eval_time_s"],
            # RM-c tidak melatih apa pun: nol parameter, nol waktu latih.
            "trainable_params": 0,
            "train_time_s": 0.0,
            "catatan": note,
        }

        if eval_test:
            test_metrics, test_extras = evaluator.evaluate(config, split="test")
            row.update(self._metric_columns(test_metrics, prefix="test"))
            self._plot_run_test(
                "rmc",
                run_id,
                features.labels["test"],
                test_extras["preds"],
                test_extras["p_judi"],
            )

        row = run_logger.log(row)

        promoted = self.best.update(
            "rmc",
            {
                "run_id": run_id,
                "config": config.model_dump(),
                "val_f1_macro": metrics["f1_macro"],
                "train_time_s": 0.0,
                "trainable_params": 0,
                "model_name": self.model_name,
            },
        )
        if promoted:
            torch.save(
                {
                    "config": config.model_dump(),
                    "run_id": run_id,
                    "val_f1_macro": metrics["f1_macro"],
                    "model_name": self.model_name,
                },
                self.checkpoint_dir / "rmc_best.pt",
            )

        logger.info("[rmc] RUN #%d val F1-macro %.4f", run_id, metrics["f1_macro"])
        return row

    # ------------------------------------------------------------------
    # Bagian benchmark final
    # ------------------------------------------------------------------

    def _final_test_metrics(
        self,
        model,
        head,
        rmc_config,
    ) -> dict[str, dict[str, float | int]]:
        features = self.features
        test_labels = features.labels["test"]

        trainer = RMATrainer(self.data)
        metrics_a, truths, predictions, positives = trainer.evaluate_test(model)
        self._plot_final("rma", truths, predictions, positives)

        with torch.no_grad():
            logits = head(torch.tensor(features.embeddings["test"], device=self.device))
        p_bert = softmax(logits.cpu().numpy())
        metrics_b = self.evaluator.metrics(test_labels, p_bert.argmax(1))
        self._plot_final("rmb", test_labels, p_bert.argmax(1), p_bert[:, 1])

        metrics_c, extras_c = RMCEvaluator(features, head, self.device).evaluate(
            rmc_config, split="test"
        )
        self._plot_final("rmc", test_labels, extras_c["preds"], extras_c["p_judi"])

        return {"rma": metrics_a, "rmb": metrics_b, "rmc": metrics_c}

    def _inference_benchmark(self, model, encoder, head, rmc_config) -> pd.DataFrame:
        features = self.features
        dataset = GamblingCommentDataset.from_frame(
            self.data.test, self.data.tokenizer, max_length=self.data.max_length
        )
        sample = next(iter(DataLoader(dataset, batch_size=1)))

        inputs = {
            "input_ids": sample["input_ids"].to(self.device),
            "attention_mask": sample["attention_mask"].to(self.device),
        }
        token_type = sample.get("token_type_ids")
        inputs["token_type_ids"] = (
            token_type.to(self.device) if token_type is not None else None
        )
        query = features.embeddings["test"][:1]

        classifier = RACClassifier(
            alpha=rmc_config.alpha, k=rmc_config.k, weighting=rmc_config.weighting
        ).fit(features.embeddings["train"], features.labels["train"])

        use_amp = self.device.type == "cuda"

        def infer_rma():
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                return model(**inputs).logits

        def infer_rmb():
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                output = encoder(**inputs)
                pooled = mean_pool(output.last_hidden_state, inputs["attention_mask"])
                return head(pooled.float())

        def infer_rmc():
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                output = encoder(**inputs)
                pooled = mean_pool(output.last_hidden_state, inputs["attention_mask"])
                p_bert = softmax(head(pooled.float()).cpu().numpy())
            return classifier.predict(query, p_bert)[1]

        rows = []
        for scenario, predict in (
            ("RM-a", infer_rma),
            ("RM-b", infer_rmb),
            ("RM-c", infer_rmc),
        ):
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.profiler.reset_peak_memory()
            latency = self.profiler.measure_latency(predict)
            memory = self.profiler.peak_gpu_memory_mb()
            rows.append(
                {
                    "scenario": scenario,
                    "infer_latency_ms": round(latency, 4),
                    "infer_peak_gpu_mem_mb": round(memory, 1),
                }
            )
            logger.info(
                "[final] %s: %.3f ms/sampel | peak %.0f MB", scenario, latency, memory
            )

        frame = pd.DataFrame(rows)
        write_csv(self.out_dir / "metrics" / "inference_benchmark.csv", frame)
        return frame

    def _comparison_table(
        self,
        test_metrics: dict[str, dict[str, float | int]],
        benchmark: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        for scenario in SCENARIOS:
            champion = self.best.get(scenario)
            metrics = test_metrics[scenario]
            rows.append(
                {
                    "model": DISPLAY_NAMES[scenario],
                    "config": json.dumps(champion["config"]),
                    "test_f1_macro": round(metrics["f1_macro"], METRIC_PRECISION),
                    "test_acc": round(metrics["accuracy"], METRIC_PRECISION),
                    "test_f1_judi": round(metrics["f1_class1"], METRIC_PRECISION),
                    "test_precision_judi": round(
                        metrics["precision_class1"], METRIC_PRECISION
                    ),
                    "test_recall_judi": round(metrics["recall_class1"], METRIC_PRECISION),
                    "val_f1_macro": round(champion["val_f1_macro"], METRIC_PRECISION),
                    "trainable_params": champion.get("trainable_params"),
                    "train_time_s": champion.get("train_time_s"),
                }
            )

        comparison = pd.DataFrame(rows).merge(
            benchmark.rename(columns={"scenario": "model"}), on="model", how="left"
        )
        write_csv(self.out_dir / "metrics" / "final_comparison.csv", comparison)
        logger.info("[final] perbandingan:\n%s", comparison.to_string(index=False))
        return comparison

    def _success_criteria(
        self, test_metrics: dict[str, dict[str, float | int]]
    ) -> pd.DataFrame:
        baseline = self.best.get("rma")
        baseline_f1 = float(test_metrics["rma"]["f1_macro"])
        baseline_params = baseline.get("trainable_params")
        baseline_time = baseline.get("train_time_s")

        rows = []
        for scenario in ("rmb", "rmc"):
            champion = self.best.get(scenario)
            gap_pp = (baseline_f1 - float(test_metrics[scenario]["f1_macro"])) * 100
            row: dict[str, object] = {
                "model": DISPLAY_NAMES[scenario],
                "f1_gap_pp": round(gap_pp, 2),
                "lolos_f1_gap<=3pp": bool(gap_pp <= MAX_F1_GAP_PP),
            }

            if baseline_params:
                reduction = (1 - (champion.get("trainable_params") or 0) / baseline_params) * 100
                row["param_reduction_pct"] = round(reduction, 4)
                row["lolos_param>=90%"] = bool(reduction >= MIN_PARAM_REDUCTION_PCT)

            if baseline_time and champion.get("train_time_s") is not None:
                reduction = (1 - champion["train_time_s"] / baseline_time) * 100
                row["train_time_s"] = champion["train_time_s"]
                row["rma_train_time_s"] = baseline_time
                row["time_reduction_pct"] = round(reduction, 2)
                row["lolos_waktu>=50%"] = bool(reduction >= MIN_TIME_REDUCTION_PCT)

            passed = sum(
                1
                for key in ("lolos_f1_gap<=3pp", "lolos_param>=90%", "lolos_waktu>=50%")
                if row.get(key)
            )
            row["kriteria_terpenuhi"] = f"{passed}/3"
            row["kompetitif(>=2/3)"] = bool(passed >= MIN_CRITERIA_PASSED)
            rows.append(row)

        criteria = pd.DataFrame(rows)
        write_csv(self.out_dir / "metrics" / "success_criteria.csv", criteria)
        logger.info("[final] kriteria sukses:\n%s", criteria.to_string(index=False))
        return criteria

    # ------------------------------------------------------------------
    # Pembantu
    # ------------------------------------------------------------------

    @staticmethod
    def _as_requests(
        requests: list[RunRequest | dict[str, object]],
    ) -> list[RunRequest]:
        return [
            item if isinstance(item, RunRequest) else RunRequest.model_validate(item)
            for item in requests
        ]

    @staticmethod
    def _effective_columns(scenario: str, frame: pd.DataFrame) -> list[str]:
        if scenario == "rma" and "micro_batch_eff" in frame.columns:
            return ["micro_batch_eff"]
        return []

    @staticmethod
    def _normalise(values: dict[str, object]) -> tuple[tuple[str, object], ...]:
        normalised: list[tuple[str, object]] = []
        for name in sorted(values):
            value = values[name]
            if isinstance(value, bool):
                normalised.append((name, value))
            elif isinstance(value, (int, float)):
                normalised.append((name, round(float(value), SIGNATURE_PRECISION)))
            else:
                normalised.append((name, str(value)))
        return tuple(normalised)

    @staticmethod
    def _metric_columns(
        metrics: dict[str, float | int],
        prefix: str = "val",
    ) -> dict[str, float]:
        if prefix == "val":
            return {
                "val_f1_macro": round(metrics["f1_macro"], METRIC_PRECISION),
                "val_acc": round(metrics["accuracy"], METRIC_PRECISION),
                "val_precision_macro": round(metrics["precision_macro"], METRIC_PRECISION),
                "val_recall_macro": round(metrics["recall_macro"], METRIC_PRECISION),
                "val_f1_judi": round(metrics["f1_class1"], METRIC_PRECISION),
            }
        return {
            f"{prefix}_{key}": round(float(metrics[key]), METRIC_PRECISION)
            for key in RUN_METRIC_KEYS
            if key in metrics
        }

    def _load_checkpoint(self, filename: str) -> dict[str, object]:
        path = self.checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint {filename} tidak ada di {self.checkpoint_dir}; "
                "jalankan skenario terkait lebih dulu"
            )
        return torch.load(path, map_location=self.device, weights_only=True)

    def _load_rma_model(self):
        checkpoint = self._load_checkpoint("rma_best.pt")
        model = build_finetune_model(
            self.data.tokenizer, model_name=self.model_name
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model

    def _load_best_head(self) -> tuple[torch.nn.Module, dict[str, object]]:
        checkpoint = self._load_checkpoint("rmb_best.pt")
        config = dict(checkpoint["config"])

        # RM-c mewarisi encoder dari head RM-b. Memakai head yang dilatih di atas
        # embedding encoder lain akan menghasilkan angka yang tampak wajar tapi
        # tidak berarti apa-apa, jadi ketidakcocokan ditolak di sini.
        source_encoder = checkpoint.get("model_name")
        if source_encoder is not None and source_encoder != self.model_name:
            raise ValueError(
                f"encoder tidak cocok: head RM-b dilatih dengan {source_encoder!r} "
                f"tapi kampanye ini memakai {self.model_name!r}; jalankan RM-b ulang "
                "dengan encoder yang sama"
            )

        head = build_head(
            config["head_arch"],
            hidden_size=int(checkpoint["hidden_size"]),
            dropout=float(config["dropout"]),
            hidden_dim=int(config.get("hidden_dim", 256)),
        )
        head.load_state_dict(checkpoint["head_state"])
        head.to(self.device).eval()
        return head, config

    def _plot_run_test(self, scenario, run_id, truths, predictions, positives) -> None:
        title = f"{DISPLAY_NAMES[scenario]} run#{run_id} (test)"
        self.reporter.confusion_matrix(
            truths, predictions, f"{scenario}_run{run_id}_confusion.png", title=title
        )
        self.reporter.pr_curve(
            truths, positives, f"{scenario}_run{run_id}_pr.png", title=title
        )

    def _plot_champion(self, scenario, run_id, truths, predictions, positives) -> None:
        title = f"{DISPLAY_NAMES[scenario]} juara saat ini (run#{run_id}, val)"
        self.reporter.confusion_matrix(
            truths, predictions, f"{scenario}_best_val_confusion.png", title=title
        )
        self.reporter.pr_curve(
            truths, positives, f"{scenario}_best_val_pr.png", title=title
        )

    def _plot_final(self, scenario, truths, predictions, positives) -> None:
        title = f"{DISPLAY_NAMES[scenario]} final (test)"
        self.reporter.confusion_matrix(
            truths, predictions, f"final_{scenario}_confusion.png", title=title
        )
        self.reporter.pr_curve(
            truths, positives, f"final_{scenario}_pr.png", title=title
        )

    def _plot_rmb_champion(self, config, result, run_id: int) -> None:
        head = build_head(
            config.head_arch,
            hidden_size=result.extras["hidden_size"],
            dropout=config.dropout,
            hidden_dim=config.hidden_dim,
        )
        head.load_state_dict(result.best_state)
        head.to(self.device).eval()

        with torch.no_grad():
            logits = head(
                torch.tensor(self.features.embeddings["val"], device=self.device)
            )
            predictions = logits.argmax(1).cpu().numpy()
            positives = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()

        self._plot_champion(
            "rmb", run_id, self.features.labels["val"], predictions, positives
        )
