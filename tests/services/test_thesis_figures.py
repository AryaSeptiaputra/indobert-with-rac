"""Test pembangkit Gambar Bab 4 di atas log kampanye sintetis."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.services.thesis_figures import ThesisFigureBuilder, format_number, format_scientific

LRS = (1e-5, 2e-5, 3e-5, 5e-5)
EPOCHS = (3, 5, 8)
HIDDEN = (128, 256, 512, 768, 1024)
ALPHAS = tuple(round(a * 0.1, 1) for a in range(11))
KS = (1, 3, 5, 10, 20, 50)
RMC_CHAMPION_HEAD = 4
RMB_CHAMPION_HEAD = 6


def tulis_log_kampanye(out_dir, final: bool = True) -> None:
    rng = np.random.default_rng(0)

    rma = [
        {"run_id": i + 1, "batch_id": "rma_tahap1_grid", "lr": lr, "epochs": ep, "batch": b,
         "val_f1_macro": 0.965 + 0.01 * rng.random(), "trainable_params": 109_485_314,
         "train_time_s": 1100.0, "peak_mem_mb": 2339.0}
        for i, (b, lr, ep) in enumerate((b, lr, ep) for b in (16, 32) for lr in LRS for ep in EPOCHS)
    ]
    pd.DataFrame(rma).to_csv(out_dir / "runs_rma.csv", index=False)

    rmb = [{"run_id": 1, "batch_id": "rmb_tuning_grid", "head_arch": "linear", "hidden_dim": 256,
            "val_f1_macro": 0.93, "trainable_params": 1_538}]
    rmb += [{"run_id": i + 2, "batch_id": "rmb_tuning_grid" if h <= 512 else "rmb_tuning_grid_stage1b",
             "head_arch": "mlp", "hidden_dim": h, "val_f1_macro": 0.95 + 0.002 * i,
             "trainable_params": 768 * h + h + 2 * h + 2} for i, h in enumerate(HIDDEN)]
    rmb += [{"run_id": 7 + i, "batch_id": "rmb_tuning_grid_stage2", "head_arch": "mlp",
             "hidden_dim": 1024, "val_f1_macro": 0.955 + 0.001 * rng.random(),
             "trainable_params": 789_506} for i in range(21)]
    rmb_frame = pd.DataFrame(rmb).assign(
        train_time_s=lambda f: 40.0 + f["run_id"], peak_mem_mb=30.0,
        extract_peak_mem_mb=850.0, train_peak_mem_mb=850.0,
    )
    rmb_frame.loc[rmb_frame["run_id"] == RMB_CHAMPION_HEAD, "val_f1_macro"] = 0.97
    rmb_frame.to_csv(out_dir / "runs_rmb.csv", index=False)

    rmc = []
    for _, head in rmb_frame.iterrows():
        for alpha in ALPHAS:
            for k in KS:
                bump = 0.004 * np.sin(np.pi * alpha) * (0.97 - head["val_f1_macro"] + 0.01) / 0.03
                rmc.append({"run_id": len(rmc) + 1, "rmb_run_id": head["run_id"], "alpha": alpha,
                            "k": k, "weighting": "similarity", "val_f1_rmb": head["val_f1_macro"],
                            "val_f1_macro": head["val_f1_macro"] + bump + 0.0005 * rng.standard_normal() * (alpha > 0),
                            "val_f1_judi": 0.9})
    pd.DataFrame(rmc).to_csv(out_dir / "runs_rmc.csv", index=False)

    best = {
        "rma": {"run_id": 14, "config": {"lr": 2e-5, "epochs": 5, "batch": 32}, "val_f1_macro": 0.975},
        "rmb": {"run_id": RMB_CHAMPION_HEAD, "val_f1_macro": 0.97},
        "rmc": {"run_id": 100, "val_f1_macro": 0.972,
                "config": {"rmb_run_id": RMC_CHAMPION_HEAD, "alpha": 0.2, "k": 5, "weighting": "similarity"}},
    }
    (out_dir / "best.json").write_text(json.dumps(best), encoding="utf-8")

    if not final:
        return
    metrics = out_dir / "metrics"
    metrics.mkdir()
    labels = np.r_[np.zeros(1150, int), np.ones(255, int)]
    predictions = pd.DataFrame({"label": labels})
    for scenario, misses in (("rma", 6), ("rmb", 14), ("rmc", 12)):
        pred = labels.copy()
        pred[-misses:] = 0
        pred[:3] = 1
        predictions[f"pred_{scenario}"] = pred
    predictions.to_csv(metrics / "final_predictions.csv", index=False)
    pd.DataFrame([
        ("RM-a", "encoder", 5.6), ("RM-a", "classification head", 0.05),
        ("RM-b", "encoder", 5.5), ("RM-b", "classification head", 0.06),
        ("RM-c", "encoder", 5.5), ("RM-c", "classification head", 0.06), ("RM-c", "retrieval dan fusi", 0.12),
    ], columns=["scenario", "component", "latency_ms"]).to_csv(metrics / "latency_breakdown.csv", index=False)
    pd.DataFrame({
        "model": ["RM-a", "RM-b", "RM-c"], "test_f1_macro": [0.9745, 0.9580, 0.9630],
        "train_time_s": [1100.0, 45.0, 44.0], "infer_latency_ms": [5.7, 5.6, 5.8],
    }).to_csv(metrics / "final_comparison.csv", index=False)


@pytest.fixture
def builder(tmp_path):
    tulis_log_kampanye(tmp_path)
    return ThesisFigureBuilder(tmp_path)


class TestFormatAngka:
    def test_koma_desimal_dan_titik_ribuan(self) -> None:
        assert format_number(1234.567, 2) == "1.234,57"

    def test_tanda_plus(self) -> None:
        assert format_number(0.214, 2, sign=True) == "+0,21"
        assert format_number(-0.5, 1, sign=True) == "-0,5"

    def test_learning_rate_notasi_ilmiah(self) -> None:
        assert format_scientific(2e-5) == r"$2 \times 10^{-5}$"
        assert format_scientific(2.5e-4) == r"$2{,}5 \times 10^{-4}$"


class TestBangunSemua:
    def test_delapan_gambar_png_dan_pdf(self, builder) -> None:
        results = builder.build_all(skip_missing=False)
        assert list(results) == [f"4.{i}" for i in range(1, 9)]
        for paths in results.values():
            assert [path.suffix for path in paths] == [".png", ".pdf"]
            assert all(path.exists() and path.stat().st_size > 0 for path in paths)

    def test_sebelum_benchmark_final_gambar_final_dilewati(self, tmp_path) -> None:
        tulis_log_kampanye(tmp_path, final=False)
        results = ThesisFigureBuilder(tmp_path).build_all()
        assert all(isinstance(results[n], list) for n in ("4.1", "4.2", "4.3", "4.4"))
        assert all(str(results[n]).startswith("dilewati") for n in ("4.5", "4.7", "4.8"))

    def test_log_hilang_dilaporkan_bila_tidak_dilewati(self, tmp_path) -> None:
        tulis_log_kampanye(tmp_path, final=False)
        with pytest.raises(FileNotFoundError, match="05_final_benchmark"):
            ThesisFigureBuilder(tmp_path).build_all(skip_missing=False)


class TestTabelTurunan:
    def test_satu_baris_per_head_dengan_kenaikan_tidak_negatif(self, builder) -> None:
        summary = builder.rac_per_head()
        assert len(summary) == 27
        assert (summary["gain_best_pp"] >= 0).all()
        assert (summary["val_f1_best"] >= summary["val_f1_shared"] - 1e-12).all()

    def test_konfigurasi_bersama_sama_untuk_semua_head(self, builder, tmp_path) -> None:
        summary = builder.rac_per_head()
        assert summary["shared_alpha"].nunique() == summary["shared_k"].nunique() == 1

        grid = pd.read_csv(tmp_path / "runs_rmc.csv")
        means = grid.groupby(["alpha", "k"])["val_f1_macro"].mean()
        assert (summary.loc[0, "shared_alpha"], summary.loc[0, "shared_k"]) == means.idxmax()

    def test_biaya_rmc_adalah_biaya_head_yang_dipakai_juara(self, builder, tmp_path) -> None:
        costs = builder.training_costs()
        head = pd.read_csv(tmp_path / "runs_rmb.csv").set_index("run_id").loc[RMC_CHAMPION_HEAD]
        assert costs.loc["RM-c", "trainable_params"] == head["trainable_params"]
        assert costs.loc["RM-c", "train_time_s"] == head["train_time_s"]
        assert costs.loc["RM-a", "train_peak_mem_mb"] == 2339.0
