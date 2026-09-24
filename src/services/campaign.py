"""Orkestrasi kampanye tuning: menjalankan run, mencatat hasil, dan benchmark final.

Satu pemanggilan `run` sama dengan satu konfigurasi dan satu baris riwayat.
Tidak ada algoritma pencarian di sini: urutan konfigurasi ditentukan manusia,
dituliskan di `tuning_grids/`, lalu diserahkan ke `run_batch`.

Kegagalan satu konfigurasi di dalam batch diisolasi ke `runs_{scenario}_errors.csv`
supaya sisa antrean tetap jalan; kampanye lima jam tidak boleh batal karena satu
nilai yang keliru di baris ke-tiga puluh.

RM-c memilih head RM-b lewat `rmb_run_id` di konfigurasinya, sehingga satu grid
RM-c adalah seluruh head RM-b x (alpha, k) dengan fusi linear, dijalankan dan
dicatat seperti skenario lain.

Seluruh seleksi hyperparameter memakai split validation. Split test hanya dibuka
di `run_final`, sekali, untuk ketiga skenario dalam satu sesi GPU yang sama --
syarat agar angka efisiensi di Bab 4 sah dibandingkan.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from src.config import SCENARIOS, settings
from src.models.comment_dataset import GamblingCommentDataset
from src.models.heads import build_encoder, build_finetune_model, build_head, mean_pool
from src.models.schemas import CONFIG_MODELS, RMBConfig, RMCConfig, RunRequest, parse_config
from src.services.data import ExperimentData
from src.services.environment import record_tuning_session, verify_final_session
from src.services.evaluation import (
    RUN_METRIC_KEYS,
    ClassificationEvaluator,
    EfficiencyProfiler,
    paired_bootstrap_f1,
)
from src.services.features import FeatureExtractor, FeatureSet
from src.services.fusion_ablation import FusionFormulaComparator, FusionFormulaConfig
from src.services.rac import RACClassifier, softmax
from src.services.reporting import FigureReporter
from src.services.run_log import BestTracker, HistoryWriter, RunLogger
from src.services.selection import (
    build_candidates,
    load_candidates,
    rank_rmc_runs,
    rank_runs,
    write_candidates,
)
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

# Selisih F1-macro antara checkpoint yang dibangun ulang dan angka tercatat yang
# masih dianggap run yang sama; di atas ini dicatat sebagai peringatan.
REPRODUCTION_TOLERANCE_PP = 0.05

# Checkpoint RM-a yang disimpan bergulir, supaya kandidat #2 bisa dievaluasi di
# test tanpa melatih ulang (pelatihan ulang tidak menjamin kurva yang identik).
RMA_TOP_CHECKPOINTS = 3
CANDIDATES_FILE = "candidates.json"


class CampaignRunner:
    """Jalankan dan catat run tuning untuk satu folder keluaran.

    Data, tokenizer, dan fitur beku dimuat malas lalu dipakai ulang, sehingga
    menjalankan ratusan konfigurasi RM-c hanya memuat encoder sekali.

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

    @property
    def heads_dir(self) -> Path:
        """Folder state tiap head RM-b (`run_{id}.pt`), dibuat bila belum ada."""
        path = self.checkpoint_dir / "rmb_heads"
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
        if scenario == "rmc":
            parsed = self._resolve_rmc_head(parsed)
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
        dengan `micro_batch=16`. Untuk RM-c, `rmb_run_id` kosong dinormalkan ke
        nomor run juara RM-b, yaitu head yang benar-benar dipakai.

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
        if scenario == "rmc":
            parsed = self._resolve_rmc_head(parsed)
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

    def restore_rmb_heads(self) -> list[int]:
        """Latih ulang head RM-b yang belum punya state tersimpan.

        Untuk kampanye yang dijalankan sebelum tahap RM-b menyimpan setiap head.
        Head dilatih ulang dari konfigurasinya di `runs_rmb.csv` (seed sama) di
        atas fitur beku yang sama. Di mesin yang sama hasilnya identik dengan run
        asli, tetapi lintas mesin kernel CUDA berbeda dan head bisa bergeser
        beberapa persepuluh poin; bergesernya dicatat sebagai peringatan.

        Returns:
            Nomor run yang head-nya dipulihkan; kosong bila semuanya sudah ada.

        Raises:
            RuntimeError: Kalau `runs_rmb.csv` kosong.
            OSError: Kalau penulisan checkpoint gagal.
        """
        rmb_runs = read_csv(self.out_dir / "runs_rmb.csv")
        if rmb_runs.empty:
            raise RuntimeError("runs_rmb.csv kosong; jalankan tahap RM-b lebih dulu")

        trainer = RMBTrainer(self.features, self.data.class_weights, self.device)
        restored: list[int] = []
        for record in rmb_runs.to_dict("records"):
            run_id = int(record["run_id"])
            if (self.heads_dir / f"run_{run_id}.pt").exists():
                continue

            config = RMBConfig.model_validate(
                {key: record[key] for key in RMBConfig.model_fields}
            )
            result = trainer.train(config)
            reproduced_f1 = float(result.best_metrics["f1_macro"])
            drift_pp = abs(reproduced_f1 - float(record["val_f1_macro"])) * 100
            if drift_pp > REPRODUCTION_TOLERANCE_PP:
                logger.warning(
                    "Head RM-b run #%d tidak mereproduksi angka lama: %.4f vs %.4f (%.3f pp)",
                    run_id, reproduced_f1, float(record["val_f1_macro"]), drift_pp,
                )
            self._save_rmb_checkpoint(
                self.heads_dir / f"run_{run_id}.pt",
                run_id, config, result.best_state, self.features.hidden_dim, reproduced_f1,
            )
            restored.append(run_id)

        logger.info("Head RM-b dipulihkan: %d run", len(restored))
        return restored

    def _restore_rmb_champion(self) -> bool:
        champion = self.best.get("rmb")
        target = self.checkpoint_dir / "rmb_best.pt"
        if not champion or target.exists():
            return False

        head_file = self.heads_dir / f"run_{int(champion['run_id'])}.pt"
        if not head_file.exists():
            raise FileNotFoundError(f"head juara RM-b {head_file} belum ada; panggil restore_rmb_heads()")
        shutil.copy2(head_file, target)
        return True

    def _restore_rmc_champion(self) -> bool:
        champion = self.best.get("rmc")
        target = self.checkpoint_dir / "rmc_best.pt"
        if not champion or target.exists():
            return False

        config = dict(champion["config"])
        payload: dict[str, object] = {
            "config": config,
            "run_id": champion.get("run_id"),
            "val_f1_macro": champion["val_f1_macro"],
            "model_name": champion.get("model_name", self.model_name),
        }
        if config.get("rmb_run_id") is not None:
            payload.update(self._head_payload(int(config["rmb_run_id"])))
        torch.save(payload, target)
        return True

    def _restore_rma_champion(self) -> bool:
        champion = self.best.get("rma")
        target = self.checkpoint_dir / "rma_best.pt"
        if not champion or target.exists():
            return False

        config = parse_config("rma", champion["config"])
        result, _ = RMATrainer(self.data).train(config)
        reproduced_f1 = float(result.best_metrics["f1_macro"])
        drift_pp = abs(reproduced_f1 - float(champion["val_f1_macro"])) * 100
        if drift_pp > REPRODUCTION_TOLERANCE_PP:
            logger.warning(
                "Juara RM-a tidak mereproduksi angka lama: %.4f vs %.4f (%.3f pp)",
                reproduced_f1, float(champion["val_f1_macro"]), drift_pp,
            )
        torch.save(
            {
                "model_state": result.best_state,
                "config": config.model_dump(),
                "run_id": champion.get("run_id"),
                "val_f1_macro": reproduced_f1,
                "model_name": self.model_name,
            },
            target,
        )
        return True

    def restore_checkpoints(self, include_rma: bool = False) -> dict[str, object]:
        """Pulihkan checkpoint yang hilang, tanpa mengubah riwayat run maupun `best.json`.

        Checkpoint tidak ikut git (`outputs/**/checkpoints/` di-gitignore), sedangkan
        `best.json` ikut. Di clone atau instance baru, checkpoint juara hilang dan
        tidak bisa dibuat lagi dengan menjalankan ulang skenarionya: F1-nya tidak
        LEBIH TINGGI dari juara yang tercatat, sehingga tidak dipromosikan dan tidak
        disimpan. Metode ini membangunnya kembali dari konfigurasi yang tercatat.

        Yang dipulihkan: state setiap head RM-b (`rmb_heads/`), `rmb_best.pt`, dan
        `rmc_best.pt`. RM-a (`rma_best.pt`) hanya dipulihkan bila diminta karena
        melatih ulang juaranya. Checkpoint yang sudah ada tidak disentuh. Di mesin
        yang sama hasilnya identik dengan run asli; lintas mesin bisa bergeser, dan
        pergeseran di atas 0,05 pp dicatat sebagai peringatan.

        Args:
            include_rma: Latih ulang juara RM-a bila `rma_best.pt` belum ada.

        Returns:
            Dict `rmb_heads` (nomor run yang dipulihkan) serta `rmb_best`, `rmc_best`,
            dan `rma_best` (True bila dipulihkan pada pemanggilan ini).

        Raises:
            RuntimeError: Kalau `runs_rmb.csv` kosong.
            FileNotFoundError: Kalau head yang dibutuhkan juara tidak ada.
        """
        restored: dict[str, object] = {"rmb_heads": self.restore_rmb_heads()}
        restored["rmb_best"] = self._restore_rmb_champion()
        restored["rmc_best"] = self._restore_rmc_champion()
        restored["rma_best"] = self._restore_rma_champion() if include_rma else False
        logger.info("Checkpoint dipulihkan: %s", restored)
        return restored

    def decide_rmc_champion(self) -> dict[str, object]:
        """Tetapkan juara RM-c: default di head RM-b resmi, atau penantang seluruh head.

        Default adalah konfigurasi fusi terbaik pada head juara RM-b, supaya RM-c
        tetap "RM-b + RAC". Penantang adalah konfigurasi terbaik di seluruh head.
        Penantang hanya menggantikan default bila selisih F1-macro validation
        melampaui ambang seri DAN batas bawah interval bootstrap berpasangan 95%
        di atas nol. Peringkat: F1-macro, lalu F1 judi, lalu head lebih murah,
        k lebih kecil, alpha lebih kecil.

        Returns:
            Catatan keputusan; juga ditulis ke `rmc_champion_decision.json`.

        Raises:
            RuntimeError: Kalau juara RM-b atau riwayat RM-c belum ada, atau head
                RM-b resmi belum dievaluasi di grid RM-c.
        """
        official = self.best.get("rmb")
        runs = read_csv(self.out_dir / "runs_rmc.csv")
        if not official or runs.empty:
            raise RuntimeError("juara RM-b dan riwayat RM-c harus ada sebelum putusan juara RM-c")

        ranked = rank_rmc_runs(runs)
        official_head = int(official["run_id"])
        on_official = ranked[ranked["rmb_run_id"] == official_head]
        if on_official.empty:
            raise RuntimeError(f"head RM-b resmi #{official_head} belum dievaluasi di grid RM-c")
        default = on_official.iloc[0]
        challenger = ranked.iloc[0]

        if int(challenger["run_id"]) == int(default["run_id"]):
            bootstrap = None
            challenger_wins = False
        else:
            bootstrap = paired_bootstrap_f1(
                self.features.labels["val"],
                self._rmc_val_predictions(challenger),
                self._rmc_val_predictions(default),
            )
            challenger_wins = bool(
                bootstrap["observed_delta_pp"] > settings.tie_threshold_pp
                and bootstrap["ci_low_pp"] > 0.0
            )

        winner = challenger if challenger_wins else default
        head_is_official = int(winner["rmb_run_id"]) == official_head
        config = self._rmc_config_of(winner)
        payload = {
            "run_id": int(winner["run_id"]),
            "config": config,
            "val_f1_macro": float(winner["val_f1_macro"]),
            "train_time_s": 0.0,
            "trainable_params": 0,
            "head_train_time_s": float(winner["head_train_time_s"]),
            "head_trainable_params": int(winner["head_trainable_params"]),
            "head_is_official_rmb": head_is_official,
            "model_name": self.model_name,
            "decided": True,
        }
        self.best.replace("rmc", payload)
        torch.save(
            {
                "config": config,
                "run_id": payload["run_id"],
                "val_f1_macro": payload["val_f1_macro"],
                "model_name": self.model_name,
                **self._head_payload(config["rmb_run_id"]),
            },
            self.checkpoint_dir / "rmc_best.pt",
        )

        decision: dict[str, object] = {
            "decided_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rule": (
                "penantang menggantikan default bila selisih F1-macro validation > "
                f"{settings.tie_threshold_pp} pp DAN batas bawah CI95 bootstrap berpasangan > 0"
            ),
            "tie_threshold_pp": settings.tie_threshold_pp,
            "official_rmb_run_id": official_head,
            "default": self._rmc_summary(default),
            "challenger": self._rmc_summary(challenger),
            "challenger_is_default": bootstrap is None,
            "bootstrap": bootstrap,
            "winner": "penantang" if challenger_wins else "default",
            "head_is_official_rmb": head_is_official,
            "cost_note": (
                "biaya pelatihan RM-c = biaya head RM-b resmi"
                if head_is_official
                else f"biaya pelatihan RM-c = biaya head penantang #{int(winner['rmb_run_id'])}, "
                "BUKAN warisan head RM-b resmi"
            ),
        }
        write_json(self.out_dir / "rmc_champion_decision.json", decision)
        logger.info(
            "Putusan juara RM-c: %s (run #%d, head #%d)",
            decision["winner"], payload["run_id"], config["rmb_run_id"],
        )
        return decision

    def _rmc_config_of(self, row: pd.Series) -> dict[str, object]:
        return RMCConfig(
            rmb_run_id=int(row["rmb_run_id"]),
            alpha=float(row["alpha"]),
            k=int(row["k"]),
            weighting=str(row["weighting"]),
        ).model_dump()

    @staticmethod
    def _rmc_summary(row: pd.Series) -> dict[str, object]:
        return {
            "run_id": int(row["run_id"]),
            "rmb_run_id": int(row["rmb_run_id"]),
            "alpha": float(row["alpha"]),
            "k": int(row["k"]),
            "weighting": str(row["weighting"]),
            "val_f1_macro": float(row["val_f1_macro"]),
            "val_f1_judi": float(row["val_f1_judi"]),
        }

    def _rmc_val_predictions(self, row: pd.Series) -> np.ndarray:
        config = RMCConfig.model_validate(self._rmc_config_of(row))
        head, _ = self._head_from_checkpoint(
            self._load_checkpoint(f"rmb_heads/run_{config.rmb_run_id}.pt")
        )
        _, extras = RMCEvaluator(self.features, head, self.device).evaluate(config, split="val")
        return extras["preds"]

    @property
    def rma_top_dir(self) -> Path:
        """Folder checkpoint top-3 RM-a (`run_{id}.pt`), dibuat bila belum ada."""
        path = self.checkpoint_dir / "rma_top"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _keep_rma_top(
        self,
        run_id: int,
        config,
        state: dict[str, torch.Tensor],
        val_f1_macro: float,
    ) -> None:
        """Simpan checkpoint run ini bila masuk top-3 RM-a, buang yang tersingkir."""
        top = set(
            rank_runs(read_csv(self.out_dir / "runs_rma.csv"), "rma")["run_id"]
            .head(RMA_TOP_CHECKPOINTS).astype(int)
        )
        if run_id in top:
            torch.save(
                {
                    "model_state": state,
                    "config": config.model_dump(),
                    "run_id": run_id,
                    "val_f1_macro": val_f1_macro,
                    "model_name": self.model_name,
                },
                self.rma_top_dir / f"run_{run_id}.pt",
            )
        for path in self.rma_top_dir.glob("run_*.pt"):
            if int(path.stem.split("_")[1]) not in top:
                path.unlink()

    def select_candidates(self) -> dict[str, object]:
        """Tetapkan kandidat #1 dan #2 tiap skenario ke `candidates.json`.

        Dipanggil di akhir 03c, SEBELUM test dibuka. Kandidat #1 adalah juara di
        `best.json`; untuk RM-c, putusan juara harus sudah dibuat.

        Returns:
            Isi `candidates.json`, termasuk `content_sha256`.

        Raises:
            RuntimeError: Kalau ada skenario tanpa juara, atau putusan juara RM-c
                belum dibuat.
            FileExistsError: Kalau kandidat sudah pernah ditetapkan.
        """
        missing = [scenario for scenario in SCENARIOS if not self.best.get(scenario)]
        if missing:
            raise RuntimeError(f"skenario {missing} belum punya juara")
        if not self.best.get("rmc").get("decided"):
            raise RuntimeError("putusan juara RM-c belum dibuat; panggil decide_rmc_champion()")

        body = build_candidates(
            {scenario: read_csv(self.out_dir / f"runs_{scenario}.csv") for scenario in SCENARIOS},
            {scenario: int(self.best.get(scenario)["run_id"]) for scenario in SCENARIOS},
        )
        stamped = write_candidates(self.out_dir / CANDIDATES_FILE, body)
        logger.info("Kandidat ditetapkan, sha256 %s", stamped["content_sha256"])
        return stamped

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

        candidates = load_candidates(self.out_dir / CANDIDATES_FILE)
        logger.info("Benchmark final dimulai (satu sesi, device %s)", self.device)
        # Seluruh model kandidat #2 disiapkan dan diverifikasi di validation
        # SEBELUM split test dibuka.
        seconds = self._prepare_second_candidates(candidates)

        model = self._load_rma_model()
        head, head_config = self._load_best_head()
        encoder = build_encoder(self.data.tokenizer, model_name=self.model_name).to(self.device)
        rmc_head, rmc_config = self._load_rmc_predictor()

        test_metrics = self._final_test_metrics(model, head, rmc_head, rmc_config)
        self._candidate_test_table(candidates, seconds, test_metrics)
        benchmark = self._inference_benchmark(model, encoder, head, rmc_head, rmc_config)
        comparison = self._comparison_table(test_metrics, benchmark)
        criteria = self._success_criteria(test_metrics)

        self.reporter.final_inference_bar_chart()
        self.reporter.write_summary()
        return {"comparison": comparison, "benchmark": benchmark, "criteria": criteria}

    def write_hardware(self) -> dict[str, object]:
        """Catat lingkungan tuning ke `hardware.json` (dipanggil di awal 03a).

        Returns:
            Isi `hardware.json`: spesifikasi lingkungan dan daftar sesi tuning.

        Raises:
            EnvironmentMismatchError: Kalau folder ini sudah memuat hasil dari
                lingkungan lain.
        """
        return record_tuning_session(self.out_dir / "hardware.json")

    def verify_hardware(self) -> dict[str, object]:
        """Gate awal 05: lingkungan wajib identik dengan saat tuning.

        Returns:
            Isi `hardware.json` setelah sesi final dicatat.

        Raises:
            EnvironmentMismatchError: Kalau GPU, VRAM, driver, CUDA, torch,
                transformers, atau faiss berbeda. Waktu boot yang berbeda hanya
                peringatan, dicatat di `final_session`.
        """
        return verify_final_session(self.out_dir / "hardware.json")

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
        if result.best_state is not None:
            self._keep_rma_top(run_id, config, result.best_state, result.best_metrics["f1_macro"])
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
            "extract_peak_mem_mb": features.extract_peak_mem_mb,
            # Pasangan waktu latih yang mencakup ekstraksi: memori puncak fase
            # pelatihan adalah yang terbesar dari ekstraksi dan training head.
            "train_peak_mem_mb": max(result.peak_mem_mb, features.extract_peak_mem_mb),
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
        if result.best_state is not None:
            # Setiap head disimpan, bukan hanya juara, karena eksplorasi RM-c
            # menguji RAC di atas seluruh head dan harus memuat head yang persis sama.
            checkpoint = self._rmb_checkpoint(
                run_id, config, result.best_state,
                result.extras["hidden_size"], result.best_metrics["f1_macro"],
            )
            torch.save(checkpoint, self.heads_dir / f"run_{run_id}.pt")
            if promoted:
                torch.save(checkpoint, self.checkpoint_dir / "rmb_best.pt")
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
        if config.rmb_run_id is None:
            raise FileNotFoundError(
                "RM-c butuh head RM-b, tetapi juara RM-b belum ada; jalankan RM-b lebih dulu"
            )

        features = self.features
        head_record = self._rmb_record(config.rmb_run_id)
        head, head_config = self._head_from_checkpoint(
            self._load_checkpoint(f"rmb_heads/run_{config.rmb_run_id}.pt")
        )
        evaluator = RMCEvaluator(features, head, self.device)

        metrics, extras = evaluator.evaluate(config, split="val")

        head_params = int(head_record["trainable_params"])
        head_train_time_s = float(head_record["train_time_s"])
        row = {
            "scenario": "rmc",
            "batch_id": batch_id,
            "model_name": self.model_name,
            **config.model_dump(),
            "head_arch_rmb": head_config.get("head_arch"),
            "hidden_dim_rmb": head_config.get("hidden_dim"),
            "val_f1_rmb": round(float(head_record["val_f1_macro"]), METRIC_PRECISION),
            **self._metric_columns(metrics),
            "index_vectors": extras["index_vectors"],
            "index_type": extras["index_type"],
            "eval_time_s": extras["eval_time_s"],
            # RM-c sendiri tidak melatih apa pun; biaya head yang dipakainya
            # dicatat terpisah agar kriteria sukses menghitungnya.
            "trainable_params": 0,
            "train_time_s": 0.0,
            "head_trainable_params": head_params,
            "head_train_time_s": head_train_time_s,
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

        # Setelah putusan juara dibuat, juara RM-c hanya berubah lewat
        # decide_rmc_champion, bukan lewat F1 mentah run susulan.
        promoted = not self.best.get("rmc").get("decided") and self.best.update(
            "rmc",
            {
                "run_id": run_id,
                "config": config.model_dump(),
                "val_f1_macro": metrics["f1_macro"],
                "train_time_s": 0.0,
                "trainable_params": 0,
                "head_train_time_s": head_train_time_s,
                "head_trainable_params": head_params,
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
                    **self._head_payload(config.rmb_run_id),
                },
                self.checkpoint_dir / "rmc_best.pt",
            )

        logger.info(
            "[rmc] RUN #%d head RM-b #%d val F1-macro %.4f",
            run_id, config.rmb_run_id, metrics["f1_macro"],
        )
        return row

    # ------------------------------------------------------------------
    # Bagian benchmark final
    # ------------------------------------------------------------------

    def _final_test_metrics(
        self,
        model,
        head,
        rmc_head,
        rmc_config: FusionFormulaConfig,
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

        comparator = FusionFormulaComparator(
            features, rmc_head, self.device, k=rmc_config.k, weighting=rmc_config.weighting
        )
        metrics_c, extras_c = comparator.evaluate(rmc_config, split="test")
        self._plot_final("rmc", test_labels, extras_c["preds"], extras_c["p_judi"])

        if not np.array_equal(np.asarray(truths), test_labels):
            raise RuntimeError("urutan split test RM-a berbeda dengan fitur beku RM-b/RM-c")
        write_csv(
            self.out_dir / "metrics" / "final_predictions.csv",
            pd.DataFrame(
                {
                    "label": test_labels,
                    "pred_rma": np.asarray(predictions),
                    "pred_rmb": p_bert.argmax(1),
                    "pred_rmc": extras_c["preds"],
                }
            ),
        )
        return {"rma": metrics_a, "rmb": metrics_b, "rmc": metrics_c}

    def _inference_benchmark(
        self, model, encoder, head, rmc_head, rmc_config: FusionFormulaConfig
    ) -> pd.DataFrame:
        """Latency dan memori inferensi satu sampel, dipecah per komponen.

        Komponen tiap skenario dijalankan berurutan dalam SATU jalur, sama persis
        dengan jalur inferensi ujung ke ujung; latency total adalah jumlah
        komponennya. Waktu bangun dan ukuran indeks FAISS ditulis ke
        `metrics/index_stats.json`.
        """
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

        train_labels = features.labels["train"]
        classifier = RACClassifier(k=rmc_config.k, weighting=rmc_config.weighting)
        started = time.perf_counter()
        classifier.fit(features.embeddings["train"], train_labels)
        index_build_s = time.perf_counter() - started
        write_json(
            self.out_dir / "metrics" / "index_stats.json",
            {
                "index_type": classifier.index_type,
                "vectors": classifier.index_size,
                "dimension": int(features.embeddings["train"].shape[1]),
                "size_mb": round(classifier.index_bytes / 1024**2, 3),
                "build_time_s": round(index_build_s, 4),
                "source_split": "train",
            },
        )
        comparator = FusionFormulaComparator(
            features, rmc_head, self.device, k=rmc_config.k, weighting=rmc_config.weighting
        )

        use_amp = self.device.type == "cuda"

        def inference(step):
            def run(value):
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    return step(value)
            return run

        def frozen_encoder(_):
            output = encoder(**inputs)
            return mean_pool(output.last_hidden_state, inputs["attention_mask"]).float()

        pipelines = {
            "RM-a": (
                ("encoder", inference(lambda _: model.base_model(**inputs).pooler_output)),
                ("classification head", inference(lambda pooled: model.classifier(model.dropout(pooled)))),
            ),
            "RM-b": (
                ("encoder", inference(frozen_encoder)),
                ("classification head", inference(lambda pooled: head(pooled))),
            ),
            "RM-c": (
                ("encoder", inference(frozen_encoder)),
                ("classification head", inference(lambda pooled: (pooled, rmc_head(pooled).float().cpu().numpy()))),
                ("retrieval FAISS", lambda state: (state[1], *classifier.retrieve(state[0].cpu().numpy()))),
                ("fusi", lambda state: comparator.fuse(rmc_config, state[0], state[1], train_labels[state[2]])[0]),
            ),
        }

        totals, components = [], []
        for scenario, stages in pipelines.items():
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.profiler.reset_peak_memory()
            per_stage = self.profiler.measure_stages(stages)
            memory = self.profiler.peak_gpu_memory_mb()
            repeats = {"latency_warmup_runs": self.profiler.n_warmup, "latency_runs": self.profiler.n_runs}
            components.extend(
                {"scenario": scenario, "component": name, "latency_ms": round(value, 4), **repeats}
                for name, value in per_stage.items()
            )
            totals.append(
                {
                    "scenario": scenario,
                    "infer_latency_ms": round(sum(per_stage.values()), 4),
                    "infer_peak_gpu_mem_mb": round(memory, 1),
                    **repeats,
                }
            )
            logger.info(
                "[final] %s: %.3f ms/sampel | peak %.0f MB", scenario, totals[-1]["infer_latency_ms"], memory
            )

        write_csv(self.out_dir / "metrics" / "latency_breakdown.csv", pd.DataFrame(components))
        frame = pd.DataFrame(totals)
        write_csv(self.out_dir / "metrics" / "inference_benchmark.csv", frame)
        return frame

    def _prepare_second_candidates(self, candidates: dict[str, object]) -> dict[str, object]:
        """Muat model kandidat #2 tiap skenario, tanpa membuka split test.

        Checkpoint RM-a #2 diambil dari `rma_top/`. Bila tidak ada, ia dilatih
        ulang sebagai jalan terakhir dan F1 validation-nya WAJIB cocok dengan log
        (toleransi 0,05 pp); bila tidak cocok, benchmark berhenti.
        """
        prepared: dict[str, object] = {}
        for scenario, entry in candidates["scenarios"].items():
            second = entry.get("second")
            if not second:
                continue
            config = parse_config(scenario, second["config"])
            if scenario == "rma":
                prepared[scenario] = self._rma_candidate_model(second, config)
            elif scenario == "rmb":
                prepared[scenario] = self._load_rmb_head(int(second["run_id"]))
            else:
                prepared[scenario] = (self._load_rmb_head(int(config.rmb_run_id)), config)
        return prepared

    def _rma_candidate_model(self, second: dict[str, object], config):
        path = self.rma_top_dir / f"run_{int(second['run_id'])}.pt"
        model = build_finetune_model(self.data.tokenizer, model_name=self.model_name).to(self.device)
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
            return model.eval()

        logger.warning("Checkpoint RM-a #%d tidak ada; dilatih ulang lalu diverifikasi", second["run_id"])
        result, model = RMATrainer(self.data).train(config)
        drift_pp = abs(float(result.best_metrics["f1_macro"]) - float(second["val_f1_macro"])) * 100
        if drift_pp > REPRODUCTION_TOLERANCE_PP:
            raise RuntimeError(
                f"RM-a kandidat #2 (run #{second['run_id']}) tidak mereproduksi F1 validation "
                f"tercatat ({drift_pp:.3f} pp > {REPRODUCTION_TOLERANCE_PP} pp); benchmark dihentikan "
                "sebelum test dibuka"
            )
        model.load_state_dict(result.best_state)
        return model.eval()

    def _candidate_test_table(
        self,
        candidates: dict[str, object],
        seconds: dict[str, object],
        test_metrics: dict[str, dict[str, float | int]],
    ) -> pd.DataFrame:
        features = self.features
        labels = features.labels["test"]
        rows = []
        for scenario, entry in candidates["scenarios"].items():
            ranked = [(1, entry["first"], test_metrics[scenario])]
            if entry.get("second"):
                ranked.append((2, entry["second"], self._second_test_metrics(scenario, seconds[scenario], labels)))
            for rank, candidate, metrics in ranked:
                rows.append(
                    {
                        "scenario": scenario,
                        "rank": rank,
                        "run_id": candidate["run_id"],
                        "config": json.dumps(candidate["config"]),
                        "val_f1_macro": candidate["val_f1_macro"],
                        "test_f1_macro": round(float(metrics["f1_macro"]), METRIC_PRECISION),
                        "test_f1_judi": round(float(metrics["f1_class1"]), METRIC_PRECISION),
                        "candidates_sha256": candidates["content_sha256"],
                    }
                )
        frame = pd.DataFrame(rows)
        write_csv(self.out_dir / "metrics" / "candidates_test.csv", frame)
        return frame

    def _second_test_metrics(self, scenario: str, prepared, labels: np.ndarray) -> dict[str, float | int]:
        if scenario == "rma":
            metrics, *_ = RMATrainer(self.data).evaluate_test(prepared)
            return metrics
        if scenario == "rmb":
            with torch.no_grad():
                logits = prepared(torch.tensor(self.features.embeddings["test"], device=self.device))
            return self.evaluator.metrics(labels, logits.argmax(1).cpu().numpy())
        head, config = prepared
        metrics, _ = RMCEvaluator(self.features, head, self.device).evaluate(config, split="test")
        return metrics

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
                    "trainable_params": self._effective_cost(scenario)[0],
                    "train_time_s": self._effective_cost(scenario)[1],
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
            trainable_params, train_time_s = self._effective_cost(scenario)
            gap_pp = (baseline_f1 - float(test_metrics[scenario]["f1_macro"])) * 100
            row: dict[str, object] = {
                "model": DISPLAY_NAMES[scenario],
                "f1_gap_pp": round(gap_pp, 2),
                "lolos_f1_gap<=3pp": bool(gap_pp <= MAX_F1_GAP_PP),
            }

            if baseline_params:
                reduction = (1 - (trainable_params or 0) / baseline_params) * 100
                row["param_reduction_pct"] = round(reduction, 4)
                row["lolos_param>=90%"] = bool(reduction >= MIN_PARAM_REDUCTION_PCT)

            if baseline_time and train_time_s is not None:
                reduction = (1 - train_time_s / baseline_time) * 100
                row["train_time_s"] = train_time_s
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

    def _effective_cost(self, scenario: str) -> tuple[int | None, float | None]:
        """Parameter terlatih dan waktu latih yang harus dibebankan ke satu skenario.

        RM-c tidak melatih apa pun, tetapi memakai head RM-b. Biayanya adalah
        biaya head itu: yang tercatat di juara RM-c (`head_*`), atau juara RM-b
        untuk catatan lama yang belum memuatnya.
        """
        champion = self.best.get(scenario)
        if scenario != "rmc":
            return champion.get("trainable_params"), champion.get("train_time_s")

        if "head_trainable_params" in champion:
            return champion["head_trainable_params"], champion.get("head_train_time_s")

        rmb_champion = self.best.get("rmb")
        return rmb_champion.get("trainable_params"), rmb_champion.get("train_time_s")

    def _resolve_rmc_head(self, config: RMCConfig) -> RMCConfig:
        """Isi `rmb_run_id` kosong dengan juara RM-b; biarkan kosong bila belum ada."""
        champion = self.best.get("rmb")
        if config.rmb_run_id is not None or not champion:
            return config
        return config.model_copy(update={"rmb_run_id": int(champion["run_id"])})

    def _rmb_record(self, rmb_run_id: int) -> dict[str, object]:
        rmb_runs = read_csv(self.out_dir / "runs_rmb.csv")
        matches = rmb_runs[rmb_runs["run_id"] == rmb_run_id] if not rmb_runs.empty else rmb_runs
        if matches.empty:
            raise ValueError(f"head RM-b run #{rmb_run_id} tidak ada di runs_rmb.csv")
        return matches.iloc[0].to_dict()

    def _head_payload(self, rmb_run_id: int) -> dict[str, object]:
        checkpoint = self._load_checkpoint(f"rmb_heads/run_{rmb_run_id}.pt")
        return {
            "head_state": checkpoint["head_state"],
            "head_config": checkpoint["config"],
            "hidden_size": checkpoint["hidden_size"],
        }

    def _rmb_checkpoint(
        self,
        run_id: int,
        config: RMBConfig,
        head_state: dict[str, torch.Tensor],
        hidden_size: int,
        val_f1_macro: float,
    ) -> dict[str, object]:
        return {
            "head_state": head_state,
            "config": config.model_dump(),
            "run_id": run_id,
            "hidden_size": hidden_size,
            "val_f1_macro": val_f1_macro,
            "model_name": self.model_name,
        }

    def _save_rmb_checkpoint(
        self,
        path: Path,
        run_id: int,
        config: RMBConfig,
        head_state: dict[str, torch.Tensor],
        hidden_size: int,
        val_f1_macro: float,
    ) -> None:
        torch.save(
            self._rmb_checkpoint(run_id, config, head_state, hidden_size, val_f1_macro), path
        )

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
                f"checkpoint {filename} tidak ada di {self.checkpoint_dir}; jalankan "
                "skenario terkait lebih dulu, atau bila best.json sudah ada (checkpoint "
                "tidak ikut git) pulihkan dengan runner.restore_checkpoints()"
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
        return self._head_from_checkpoint(self._load_checkpoint("rmb_best.pt"))

    def _load_rmb_head(self, run_id: int) -> torch.nn.Module:
        head, _ = self._head_from_checkpoint(self._load_checkpoint(f"rmb_heads/run_{run_id}.pt"))
        return head

    def _load_rmc_predictor(self) -> tuple[torch.nn.Module, FusionFormulaConfig]:
        """Head dan rumus fusi juara RM-c.

        `rmc_best.pt` membawa head yang dipakai juara. Checkpoint lama yang belum
        memuatnya memakai head juara RM-b.
        """
        checkpoint = self._load_checkpoint("rmc_best.pt")
        config = dict(checkpoint["config"])
        fusion = FusionFormulaConfig(
            formula=config.get("formula", "linear"),
            alpha=config.get("alpha"),
            k=int(config["k"]),
            weighting=config.get("weighting", "similarity"),
        )

        if "head_state" not in checkpoint:
            head, _ = self._load_best_head()
            return head, fusion

        head, _ = self._head_from_checkpoint(
            {
                "config": checkpoint["head_config"],
                "head_state": checkpoint["head_state"],
                "hidden_size": checkpoint["hidden_size"],
                "model_name": checkpoint.get("model_name"),
            }
        )
        return head, fusion

    def _head_from_checkpoint(
        self, checkpoint: dict[str, object]
    ) -> tuple[torch.nn.Module, dict[str, object]]:
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
