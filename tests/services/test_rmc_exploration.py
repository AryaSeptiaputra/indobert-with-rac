"""Test eksplorasi RM-c: sapuan head x rumus fusi, seleksi juara, dan aturan penggantian."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.heads import build_head
from src.models.schemas import RMBConfig
from src.services.fusion_ablation import FusionFormulaConfig
from src.services.rmc_exploration import (
    RMCExplorer,
    challenger_wins,
    load_exploration_grid,
    paired_bootstrap,
    pareto_front,
    select_best,
    summarize_per_formula,
    summarize_per_head,
)
from src.services.training import RMBTrainer

GRID = [
    FusionFormulaConfig("linear", alpha=0.0, k=1),
    FusionFormulaConfig("linear", alpha=0.3, k=3),
    FusionFormulaConfig("linear", alpha=0.3, k=5),
    FusionFormulaConfig("rumus1", k=5),
    FusionFormulaConfig("rumus2", k=5),
    FusionFormulaConfig("rumus3", k=5),
    FusionFormulaConfig("rumus4", alpha=0.3, k=5),
]


@pytest.fixture
def heads_and_runs(feature_set, class_weights, cpu_device):
    """Dua head kecil beserta baris `runs_rmb.csv` yang mencatat F1-macro sebenarnya."""
    trainer = RMBTrainer(feature_set, class_weights, cpu_device)
    heads, rows = {}, []
    for run_id, config in enumerate(
        [RMBConfig(head_arch="linear", epochs=3), RMBConfig(head_arch="mlp", hidden_dim=16, epochs=3)],
        start=1,
    ):
        result = trainer.train(config)
        head = build_head(
            config.head_arch,
            hidden_size=feature_set.hidden_dim,
            dropout=config.dropout,
            hidden_dim=config.hidden_dim,
        )
        head.load_state_dict(result.best_state)
        heads[run_id] = head.eval()
        rows.append(
            {
                "run_id": run_id,
                **config.model_dump(),
                "val_f1_macro": result.best_metrics["f1_macro"],
                "trainable_params": result.trainable_params,
                "train_time_s": 1.0 * run_id,
            }
        )
    return heads, pd.DataFrame(rows)


@pytest.fixture
def sweep(feature_set, cpu_device, heads_and_runs) -> pd.DataFrame:
    heads, rmb_runs = heads_and_runs
    return RMCExplorer(feature_set, cpu_device, GRID).run(rmb_runs, heads)


class TestRMCExplorer:
    def test_satu_baris_per_pasangan_head_dan_konfigurasi(self, sweep, heads_and_runs) -> None:
        _, rmb_runs = heads_and_runs
        assert len(sweep) == len(rmb_runs) * len(GRID)
        assert set(sweep["rmb_run_id"]) == set(rmb_runs["run_id"])

    def test_head_yang_dimuat_mereproduksi_angka_rmb_yang_dicatat(self, sweep) -> None:
        assert (sweep["val_f1_rmb"] - sweep["val_f1_rmb_logged"]).abs().max() < 1e-5

    def test_alpha_nol_sama_dengan_baseline_head(self, sweep) -> None:
        at_zero = sweep[(sweep["formula"] == "linear") & (sweep["alpha"] == 0.0)]
        assert (at_zero["val_f1_macro"] == at_zero["val_f1_rmb"]).all()
        assert (at_zero["gain_pp"] == 0.0).all()

    def test_gain_adalah_selisih_terhadap_baseline_dalam_poin_persentase(self, sweep) -> None:
        expected = (sweep["val_f1_macro"] - sweep["val_f1_rmb"]) * 100
        assert (sweep["gain_pp"] - expected).abs().max() < 1e-3

    def test_rumus_tanpa_alpha_dicatat_kosong_dan_alpha_adaptif_dirata_ratakan(self, sweep) -> None:
        adaptive = sweep[sweep["formula"] == "rumus2"]
        assert adaptive["alpha"].isna().all()
        assert adaptive["alpha_mean"].between(0.0, 1.0).all()
        assert sweep.loc[sweep["formula"] == "rumus1", "alpha_mean"].isna().all()

    def test_biaya_head_ikut_tercatat(self, sweep, heads_and_runs) -> None:
        _, rmb_runs = heads_and_runs
        per_run = sweep.drop_duplicates("rmb_run_id").set_index("rmb_run_id")
        assert per_run.loc[2, "head_params"] == rmb_runs.set_index("run_id").loc[2, "trainable_params"]
        assert per_run.loc[2, "head_train_time_s"] == 2.0

    def test_kandidat_kosong_ditolak(self, feature_set, cpu_device) -> None:
        with pytest.raises(ValueError, match="candidates kosong"):
            RMCExplorer(feature_set, cpu_device, [])

    def test_rmb_runs_kosong_ditolak(self, feature_set, cpu_device) -> None:
        with pytest.raises(ValueError, match="rmb_runs kosong"):
            RMCExplorer(feature_set, cpu_device, GRID).run(pd.DataFrame(), {})

    def test_head_yang_tidak_tersedia_ditolak(self, feature_set, cpu_device, heads_and_runs) -> None:
        heads, rmb_runs = heads_and_runs
        with pytest.raises(ValueError, match="tidak tersedia"):
            RMCExplorer(feature_set, cpu_device, GRID).run(rmb_runs, {1: heads[1]})


class TestLoadExplorationGrid:
    def test_membaca_alpha_kosong_sebagai_none(self, tmp_path) -> None:
        path = tmp_path / "grid.csv"
        path.write_text(
            "formula,alpha,k,weighting,catatan\n"
            "linear,0.3,5,similarity,a\n"
            "rumus2,,10,similarity,b\n",
            encoding="utf-8",
        )
        configs = load_exploration_grid(path)
        assert configs[0] == FusionFormulaConfig("linear", alpha=0.3, k=5)
        assert configs[1] == FusionFormulaConfig("rumus2", alpha=None, k=10)

    def test_berkas_hilang_ditolak(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="kosong atau tidak ditemukan"):
            load_exploration_grid(tmp_path / "tidak_ada.csv")

    def test_kolom_wajib_hilang_ditolak(self, tmp_path) -> None:
        path = tmp_path / "grid.csv"
        path.write_text("formula,alpha\nlinear,0.3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="hilang"):
            load_exploration_grid(path)

    def test_grid_rancangan_di_repo_memuat_133_konfigurasi(self) -> None:
        from src.config import PROJECT_ROOT

        configs = load_exploration_grid(PROJECT_ROOT / "tuning_grids" / "RMC_EXPLORATION_GRID.csv")
        assert len(configs) == 133
        assert len({(c.formula, c.alpha, c.k, c.weighting) for c in configs}) == 133


def make_rows(rows: list[tuple]) -> pd.DataFrame:
    """Baris hasil minimal: (head, params, waktu, rumus, alpha, k, f1, f1_judi)."""
    return pd.DataFrame(
        [
            {
                "rmb_run_id": head,
                "head_params": params,
                "head_train_time_s": time_s,
                "formula": formula,
                "alpha": alpha,
                "k": k,
                "weighting": "similarity",
                "val_f1_macro": f1,
                "val_f1_judi": judi,
                "val_f1_rmb": 0.90,
                "val_f1_judi_rmb": 0.85,
                "gain_pp": round((f1 - 0.90) * 100, 4),
                "gain_judi_pp": round((judi - 0.85) * 100, 4),
                "head_arch": "mlp",
                "hidden_dim": 8,
                "epochs": 3,
                "lr": 1e-3,
                "dropout": 0.1,
                "weight_decay": 0.01,
                "batch": 32,
            }
            for head, params, time_s, formula, alpha, k, f1, judi in rows
        ]
    )


class TestSelectBest:
    def test_f1_macro_jauh_lebih_tinggi_menang_walau_head_lebih_mahal(self) -> None:
        frame = make_rows(
            [(1, 100, 5, "linear", 0.3, 5, 0.95, 0.9), (2, 900, 9, "rumus2", None, 5, 0.97, 0.9)]
        )
        assert select_best(frame)["rmb_run_id"] == 2

    def test_seri_memilih_head_lebih_murah(self) -> None:
        frame = make_rows(
            [(1, 100, 5, "rumus4", 0.3, 5, 0.9490, 0.9), (2, 900, 9, "linear", 0.3, 5, 0.9500, 0.9)]
        )
        assert select_best(frame, tie_threshold_pp=0.15)["rmb_run_id"] == 1

    def test_seri_dan_head_sama_memilih_fusi_linear(self) -> None:
        frame = make_rows(
            [(1, 100, 5, "rumus3", None, 5, 0.9500, 0.9), (1, 100, 5, "linear", 0.3, 5, 0.9495, 0.9)]
        )
        assert select_best(frame, tie_threshold_pp=0.15)["formula"] == "linear"

    def test_seri_penuh_mendahulukan_k_terkecil(self) -> None:
        frame = make_rows(
            [
                (1, 100, 5, "linear", 0.4, 3, 0.95, 0.9),
                (1, 100, 5, "linear", 0.2, 10, 0.95, 0.9),
                (1, 100, 5, "linear", 0.2, 5, 0.95, 0.9),
            ]
        )
        best = select_best(frame, tie_threshold_pp=0.15)
        assert (best["alpha"], best["k"]) == (0.4, 3)

    def test_k_sama_memilih_alpha_terkecil(self) -> None:
        frame = make_rows(
            [
                (1, 100, 5, "linear", 0.4, 5, 0.95, 0.9),
                (1, 100, 5, "linear", 0.2, 5, 0.95, 0.9),
                (1, 100, 5, "linear", 0.3, 5, 0.95, 0.9),
            ]
        )
        assert select_best(frame, tie_threshold_pp=0.15)["alpha"] == 0.2

    def test_selisih_waktu_di_bawah_satu_detik_tidak_menentukan(self) -> None:
        """Derau pengukuran 9,03 s versus 9,06 s tidak boleh memilih pemenang."""
        frame = make_rows(
            [(1, 100, 9.06, "linear", 0.3, 5, 0.9500, 0.9), (2, 100, 9.03, "linear", 0.3, 3, 0.9500, 0.9)]
        )
        assert select_best(frame, tie_threshold_pp=0.15)["k"] == 3

    def test_di_luar_ambang_seri_f1_menentukan(self) -> None:
        frame = make_rows(
            [(1, 100, 5, "linear", 0.3, 5, 0.9400, 0.9), (2, 900, 9, "linear", 0.3, 5, 0.9500, 0.9)]
        )
        assert select_best(frame, tie_threshold_pp=0.15)["rmb_run_id"] == 2


class TestSummaries:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return make_rows(
            [
                (1, 100, 5, "linear", 0.0, 1, 0.90, 0.85),
                (1, 100, 5, "linear", 0.3, 5, 0.94, 0.89),
                (1, 100, 5, "rumus2", None, 5, 0.92, 0.87),
                (2, 900, 9, "linear", 0.0, 1, 0.90, 0.85),
                (2, 900, 9, "linear", 0.3, 5, 0.9005, 0.85),
                (2, 900, 9, "rumus2", None, 5, 0.91, 0.86),
            ]
        )

    def test_satu_baris_per_head(self, frame: pd.DataFrame) -> None:
        assert list(summarize_per_head(frame)["rmb_run_id"]) == [1, 2]

    def test_gain_di_bawah_ambang_seri_tidak_dihitung_membantu(self, frame: pd.DataFrame) -> None:
        per_head = summarize_per_head(frame).set_index("rmb_run_id")
        assert bool(per_head.loc[1, "rac_helps"]) is True
        assert per_head.loc[1, "best_formula"] == "linear"
        assert bool(per_head.loc[2, "rac_helps"]) is True  # rumus2 memberi +1,0 pp

    def test_biaya_head_dan_tanda_pareto_dilaporkan(self, frame: pd.DataFrame) -> None:
        per_head = summarize_per_head(frame).set_index("rmb_run_id")
        assert per_head.loc[1, "head_params"] == 100
        # Head 2 lebih mahal DAN F1 terbaiknya lebih rendah (0,91 vs 0,94): didominasi.
        assert bool(per_head.loc[1, "on_pareto"]) is True
        assert bool(per_head.loc[2, "on_pareto"]) is False

    def test_satu_baris_per_rumus_dengan_konfigurasi_seragam(self, frame: pd.DataFrame) -> None:
        per_formula = summarize_per_formula(frame).set_index("formula")
        assert set(per_formula.index) == {"linear", "rumus2"}
        assert per_formula.loc["rumus2", "shared_k"] == 5
        assert per_formula.loc["rumus2", "n_heads"] == 2
        assert np.isnan(per_formula.loc["rumus2", "shared_alpha"])
        assert per_formula.loc["linear", "shared_alpha"] == 0.3

    def test_rata_rata_seragam_dihitung_lintas_head(self, frame: pd.DataFrame) -> None:
        per_formula = summarize_per_formula(frame).set_index("formula")
        assert per_formula.loc["linear", "shared_mean_val_f1_macro"] == pytest.approx((0.94 + 0.9005) / 2)


class TestParetoFront:
    def test_baris_yang_didominasi_tidak_masuk(self) -> None:
        frame = pd.DataFrame(
            {"f1": [0.95, 0.96, 0.94], "params": [100, 100, 100], "time": [5, 5, 5]}
        )
        on_front = pareto_front(frame, maximize=["f1"], minimize=["params", "time"])
        assert on_front.tolist() == [False, True, False]

    def test_trade_off_performa_dan_biaya_sama_sama_bertahan(self) -> None:
        frame = pd.DataFrame({"f1": [0.95, 0.97], "params": [100, 900], "time": [5, 9]})
        assert pareto_front(frame, ["f1"], ["params", "time"]).all()

    def test_kembar_persis_tidak_saling_mendominasi(self) -> None:
        frame = pd.DataFrame({"f1": [0.95, 0.95], "params": [100, 100]})
        assert pareto_front(frame, ["f1"], ["params"]).all()


class TestPairedBootstrap:
    def test_prediksi_identik_tidak_memberi_selisih(self, rng) -> None:
        y = rng.integers(0, 2, 200)
        delta, low, high = paired_bootstrap(y, y.copy(), y.copy(), n_boot=200)
        assert (delta, low, high) == (0.0, 0.0, 0.0)

    def test_penantang_yang_jauh_lebih_baik_lolos(self, rng) -> None:
        y = rng.integers(0, 2, 400)
        good = y.copy()
        bad = np.where(rng.random(400) < 0.25, 1 - y, y)
        delta, low, _ = paired_bootstrap(y, good, bad, n_boot=500)
        assert delta > 0 and low > 0

    def test_selisih_dua_sampel_tidak_lolos(self, rng) -> None:
        """Skenario nyata sapuan: penantang unggul di 2 dari 1.402 sampel."""
        y = rng.integers(0, 2, 1402)
        incumbent = np.where(rng.random(1402) < 0.03, 1 - y, y)
        challenger = incumbent.copy()
        wrong = np.where(incumbent != y)[0][:2]
        challenger[wrong] = y[wrong]
        delta, low, high = paired_bootstrap(y, challenger, incumbent, n_boot=1000)
        assert delta > 0
        assert low <= 0 <= high

    def test_hasil_deterministik_untuk_seed_sama(self, rng) -> None:
        y = rng.integers(0, 2, 100)
        a = np.where(rng.random(100) < 0.1, 1 - y, y)
        b = np.where(rng.random(100) < 0.2, 1 - y, y)
        assert paired_bootstrap(y, a, b, n_boot=100, seed=7) == paired_bootstrap(y, a, b, n_boot=100, seed=7)

    def test_panjang_tidak_sama_ditolak(self) -> None:
        with pytest.raises(ValueError, match="sepanjang sama"):
            paired_bootstrap(np.zeros(3), np.zeros(4), np.zeros(3))


class TestChallengerWins:
    def test_butuh_selisih_di_atas_ambang_dan_interval_di_atas_nol(self) -> None:
        assert challenger_wins(0.30, 0.05, tie_threshold_pp=0.15) is True

    def test_selisih_di_bawah_ambang_seri_kalah(self) -> None:
        assert challenger_wins(0.10, 0.05, tie_threshold_pp=0.15) is False

    def test_interval_yang_melewati_nol_kalah_walau_selisih_besar(self) -> None:
        assert challenger_wins(0.50, -0.20, tie_threshold_pp=0.15) is False

    def test_ambang_bawaan_dari_settings(self) -> None:
        assert challenger_wins(0.16, 0.01) is True
        assert challenger_wins(0.14, 0.01) is False
