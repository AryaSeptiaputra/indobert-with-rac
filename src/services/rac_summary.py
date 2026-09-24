"""Ringkasan kontribusi RAC per head RM-b dari `runs_rmc.csv`.

Dipakai bersama oleh Gambar 4.3-4.4, Tabel 4.11-4.12, dan Tabel L.1 supaya
semuanya membaca angka yang sama. F1 "tanpa RAC" diambil dari baris `alpha=0`
di grid RM-c, yaitu head yang DIMUAT dari checkpoint, bukan angka yang tercatat
saat head itu dilatih.
"""

from __future__ import annotations

import pandas as pd

GRID_WEIGHTING = "similarity"
PP = 100.0


def rank_rmc_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Urutkan run RM-c: F1-macro, F1 judi, head lebih murah, k dan alpha lebih kecil."""
    return runs.sort_values(
        ["val_f1_macro", "val_f1_judi", "head_trainable_params", "k", "alpha"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def rmc_grid(runs_rmc: pd.DataFrame) -> pd.DataFrame:
    """Baris grid `alpha x k` (pembobotan `similarity`) beserta baseline dan gain per head.

    Args:
        runs_rmc: Isi `runs_rmc.csv`.

    Returns:
        Baris grid dengan kolom tambahan `f1_no_rac`, `f1_judi_no_rac`, `gain_pp`,
        dan `gain_judi_pp`.

    Raises:
        ValueError: Kalau ada head tanpa baris `alpha=0`.
    """
    grid = runs_rmc[runs_rmc["weighting"] == GRID_WEIGHTING].copy()
    grid["rmb_run_id"] = grid["rmb_run_id"].astype(int)

    baseline = (
        grid[grid["alpha"] == 0.0]
        .sort_values(["rmb_run_id", "k"])
        .groupby("rmb_run_id")[["val_f1_macro", "val_f1_judi"]]
        .first()
        .rename(columns={"val_f1_macro": "f1_no_rac", "val_f1_judi": "f1_judi_no_rac"})
    )
    missing = sorted(set(grid["rmb_run_id"]) - set(baseline.index))
    if missing:
        raise ValueError(f"head RM-b {missing} tidak punya baris alpha=0 di grid RM-c")

    grid = grid.join(baseline, on="rmb_run_id")
    grid["gain_pp"] = (grid["val_f1_macro"] - grid["f1_no_rac"]) * PP
    grid["gain_judi_pp"] = (grid["val_f1_judi"] - grid["f1_judi_no_rac"]) * PP
    return grid


def shared_config(grid: pd.DataFrame) -> tuple[float, int]:
    """(alpha, k) dengan rata-rata gain tertinggi di seluruh head; seri: alpha lalu k lebih kecil."""
    means = grid.groupby(["alpha", "k"])["gain_pp"].mean().reset_index()
    best = means.sort_values(["gain_pp", "alpha", "k"], ascending=[False, True, True]).iloc[0]
    return float(best["alpha"]), int(best["k"])


def rac_per_head(runs_rmc: pd.DataFrame, runs_rmb: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per head: tanpa RAC, konfigurasi terbaik, dan konfigurasi bersama.

    Konfigurasi terbaik per head diurutkan menurut F1-macro, lalu F1 judi, lalu
    k dan alpha lebih kecil.

    Args:
        runs_rmc: Isi `runs_rmc.csv`.
        runs_rmb: Isi `runs_rmb.csv`, untuk arsitektur dan hyperparameter head.

    Returns:
        DataFrame per head.
    """
    grid = rmc_grid(runs_rmc)
    alpha_shared, k_shared = shared_config(grid)

    best = (
        grid.sort_values(
            ["val_f1_macro", "val_f1_judi", "k", "alpha"], ascending=[False, False, True, True]
        )
        .groupby("rmb_run_id", sort=False)
        .head(1)
        .set_index("rmb_run_id")
    )
    at_shared = grid[(grid["alpha"] == alpha_shared) & (grid["k"] == k_shared)].set_index("rmb_run_id")

    summary = pd.DataFrame(
        {
            "f1_no_rac": best["f1_no_rac"],
            "f1_judi_no_rac": best["f1_judi_no_rac"],
            "best_alpha": best["alpha"],
            "best_k": best["k"],
            "f1_best": best["val_f1_macro"],
            "gain_best_pp": best["gain_pp"],
            "gain_judi_best_pp": best["gain_judi_pp"],
            "f1_shared": at_shared["val_f1_macro"],
            "gain_shared_pp": at_shared["gain_pp"],
        }
    )
    summary["shared_alpha"] = alpha_shared
    summary["shared_k"] = k_shared

    heads = runs_rmb.set_index("run_id")[["head_arch", "hidden_dim", "lr", "epochs", "trainable_params"]]
    summary = summary.join(heads)
    return summary.rename_axis("rmb_run_id").reset_index()


def head_group(head_arch: str, hidden_dim: int) -> str:
    """Kelompok head menurut arsitektur, ditetapkan sebelum melihat hasil."""
    if head_arch == "linear":
        return "Linear"
    if int(hidden_dim) < 1024:
        return "MLP hidden dim < 1024"
    return f"MLP hidden dim = {int(hidden_dim)}"


__all__ = ["head_group", "rac_per_head", "rank_rmc_runs", "rmc_grid", "shared_config"]
