"""Test penetapan kandidat #1 dan #2 untuk uji sensitivitas (Tabel 4.16)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from src.services.selection import (
    NO_TIED_CANDIDATE,
    build_candidates,
    load_candidates,
    select_second,
    write_candidates,
)
from tests.services.test_campaign import runner, tiga_head  # noqa: F401


def rmb_runs() -> pd.DataFrame:
    base = {"head_arch": "mlp", "hidden_dim": 1024, "lr": 1e-3, "dropout": 0.1,
            "weight_decay": 0.01, "batch": 32, "seed": 42}
    rows = [
        {"run_id": 1, "epochs": 10, "best_epoch": 6, "val_f1_macro": 0.9700, "val_f1_judi": 0.95, "train_time_s": 40.0},
        {"run_id": 2, "epochs": 20, "best_epoch": 6, "val_f1_macro": 0.9700, "val_f1_judi": 0.95, "train_time_s": 60.0},
        {"run_id": 3, "epochs": 10, "best_epoch": 7, "val_f1_macro": 0.9690, "val_f1_judi": 0.96, "train_time_s": 45.0, "lr": 5e-4},
        {"run_id": 4, "epochs": 10, "best_epoch": 5, "val_f1_macro": 0.9690, "val_f1_judi": 0.94, "train_time_s": 30.0, "dropout": 0.3},
        {"run_id": 5, "epochs": 10, "best_epoch": 5, "val_f1_macro": 0.9600, "val_f1_judi": 0.99, "train_time_s": 10.0, "dropout": 0.0},
    ]
    return pd.DataFrame([{**base, **row} for row in rows])


def rma_runs() -> pd.DataFrame:
    base = {"warmup_ratio": 0.1, "weight_decay": 0.01, "micro_batch": 32, "seed": 42, "batch": 32, "lr": 2e-5}
    rows = [
        {"run_id": 1, "epochs": 5, "best_epoch": 4, "val_f1_macro": 0.9770, "val_f1_judi": 0.96, "train_time_s": 140.0},
        {"run_id": 2, "epochs": 8, "best_epoch": 4, "val_f1_macro": 0.9765, "val_f1_judi": 0.96, "train_time_s": 220.0},
    ]
    return pd.DataFrame([{**base, **row} for row in rows])


class TestKandidatKedua:
    def test_rmb_varian_epoch_dengan_epoch_terbaik_sama_dikeluarkan(self) -> None:
        second, excluded = select_second(rmb_runs(), "rmb", 1)
        assert excluded == [2]
        assert int(second["run_id"]) == 3

    def test_pemecah_seri_f1_judi_lalu_training_time(self) -> None:
        runs = rmb_runs()
        runs.loc[runs["run_id"] == 3, "val_f1_judi"] = 0.94
        second, _ = select_second(runs, "rmb", 1)
        assert int(second["run_id"]) == 4

    def test_rma_varian_epoch_tidak_dikeluarkan(self) -> None:
        """Jadwal linear-warmup RM-a bergantung pada total langkah: bobotnya berbeda."""
        second, excluded = select_second(rma_runs(), "rma", 1)
        assert excluded == []
        assert int(second["run_id"]) == 2

    def test_di_luar_ambang_seri_tidak_ada_kandidat(self) -> None:
        runs = rmb_runs()[lambda f: f["run_id"].isin([1, 5])]
        second, _ = select_second(runs, "rmb", 1)
        assert second is None

    def test_rmc_dicari_pada_head_yang_sama(self) -> None:
        runs = pd.DataFrame([
            {"run_id": 1, "rmb_run_id": 6, "alpha": 0.2, "k": 5, "weighting": "similarity", "val_f1_macro": 0.972, "val_f1_judi": 0.95, "head_train_time_s": 40.0},
            {"run_id": 2, "rmb_run_id": 9, "alpha": 0.2, "k": 5, "weighting": "similarity", "val_f1_macro": 0.9725, "val_f1_judi": 0.99, "head_train_time_s": 30.0},
            {"run_id": 3, "rmb_run_id": 6, "alpha": 0.3, "k": 5, "weighting": "similarity", "val_f1_macro": 0.971, "val_f1_judi": 0.95, "head_train_time_s": 40.0},
        ])
        second, _ = select_second(runs, "rmc", 1)
        assert int(second["run_id"]) == 3

    def test_isi_mencatat_perbedaan_dan_catatan_tanpa_kandidat(self) -> None:
        body = build_candidates(
            {"rmb": rmb_runs(), "rma": rma_runs().head(1)}, {"rmb": 1, "rma": 1}
        )
        assert body["scenarios"]["rmb"]["differs_in"] == {"lr": [0.001, 0.0005]}
        assert body["scenarios"]["rma"]["second"] is None
        assert body["scenarios"]["rma"]["note"] == NO_TIED_CANDIDATE


class TestHashKandidat:
    def test_hash_terverifikasi_dan_perubahan_terdeteksi(self, tmp_path) -> None:
        path = tmp_path / "candidates.json"
        stamped = write_candidates(path, build_candidates({"rmb": rmb_runs()}, {"rmb": 1}))
        assert load_candidates(path)["content_sha256"] == stamped["content_sha256"]

        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["scenarios"]["rmb"]["second"]["run_id"] = 99
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="hash"):
            load_candidates(path)

    def test_kandidat_tidak_ditimpa(self, tmp_path) -> None:
        body = build_candidates({"rmb": rmb_runs()}, {"rmb": 1})
        write_candidates(tmp_path / "candidates.json", body)
        with pytest.raises(FileExistsError):
            write_candidates(tmp_path / "candidates.json", body)


class TestIntegrasiRunner:
    def test_butuh_putusan_juara_rmc(self, tiga_head) -> None:
        tiga_head.run("rmc", {"alpha": 0.2})
        tiga_head.best.data["rma"] = {"run_id": 1, "val_f1_macro": 0.9}
        with pytest.raises(RuntimeError, match="putusan juara RM-c"):
            tiga_head.select_candidates()

    def test_top3_rma_bergulir(self, runner, tmp_path) -> None:
        from src.models.schemas import RMAConfig

        rows = []
        for run_id, f1 in enumerate((0.90, 0.95, 0.93, 0.97, 0.91), start=1):
            rows.append({"run_id": run_id, "val_f1_macro": f1, "val_f1_judi": 0.9, "train_time_s": 1.0})
            pd.DataFrame(rows).to_csv(tmp_path / "runs_rma.csv", index=False)
            runner._keep_rma_top(run_id, RMAConfig(), {"w": torch.zeros(1)}, f1)

        disimpan = sorted(int(p.stem.split("_")[1]) for p in runner.rma_top_dir.glob("run_*.pt"))
        assert disimpan == [2, 3, 4]

    def test_kandidat_kedua_dievaluasi_di_test_dengan_hash(self, tiga_head, tmp_path) -> None:
        tiga_head.run_batch("rmc", [{"config": {"rmb_run_id": 2, "alpha": a, "k": 3}} for a in (0.0, 0.3)])
        body = build_candidates(
            {s: pd.read_csv(tmp_path / f"runs_{s}.csv") for s in ("rmb", "rmc")},
            {"rmb": 1, "rmc": 1},
        )
        body["scenarios"]["rmb"]["second"] = {"run_id": 3, "config": {}, "val_f1_macro": 0.5, "val_f1_judi": 0.5}
        body["scenarios"]["rmc"]["second"] = {
            "run_id": 2, "config": {"rmb_run_id": 2, "alpha": 0.3, "k": 3, "weighting": "similarity"},
            "val_f1_macro": 0.5, "val_f1_judi": 0.5,
        }
        stamped = write_candidates(tmp_path / "candidates.json", body)

        seconds = tiga_head._prepare_second_candidates(stamped)
        first_metrics = {"f1_macro": 0.9, "f1_class1": 0.8}
        table = tiga_head._candidate_test_table(stamped, seconds, {"rmb": first_metrics, "rmc": first_metrics})

        assert sorted(zip(table["scenario"], table["rank"])) == [("rmb", 1), ("rmb", 2), ("rmc", 1), ("rmc", 2)]
        assert (table["candidates_sha256"] == stamped["content_sha256"]).all()
        assert (tmp_path / "metrics" / "candidates_test.csv").exists()
