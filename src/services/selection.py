"""Peringkat run, kandidat #1 dan #2 per skenario, dan ringkasan kontribusi RAC.

Kandidat #2 ditetapkan dari validation SEBELUM test dibuka, ditulis ke
`candidates.json` bersama cap waktu dan hash isinya. Benchmark final mencatat
ulang hash itu sebagai bukti bahwa kandidat tidak berubah setelah test dibuka.

Definisi kandidat #2: konfigurasi dengan peringkat tertinggi berikutnya yang
(a) seri dengan kandidat #1 (selisih F1-macro validation <= ambang seri), dan
(b) untuk RM-b, bukan varian anggaran epoch dengan epoch terbaik yang sama.
RM-b memakai learning rate konstan dan urutan batch yang deterministik terhadap
seed, sehingga varian seperti itu menghasilkan bobot identik. Aturan (b) TIDAK
berlaku untuk RM-a: jadwal linear-warmup RM-a bergantung pada total langkah,
sehingga anggaran epoch yang berbeda menghasilkan bobot berbeda walau epoch
terbaiknya sama. Pemecah seri: F1 judi validation, lalu training time lebih
kecil. Untuk RM-c, kandidat #2 dicari di antara konfigurasi fusi pada head yang
sama dengan kandidat #1.

Ringkasan kontribusi RAC per head RM-b (`rac_per_head`, `rmc_grid`) dipakai
bersama oleh Gambar 4.3-4.4, Tabel 4.11-4.12, dan Tabel L.1 supaya semuanya
membaca angka yang sama. F1 "tanpa RAC" diambil dari baris `alpha=0` di grid
RM-c, yaitu head yang DIMUAT dari checkpoint, bukan angka yang tercatat saat
head itu dilatih.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from src.config import settings
from src.models.schemas import CONFIG_MODELS
from src.utils.io import read_json, write_json

PP = 100.0
FLOAT_TOLERANCE = 1e-9
NO_TIED_CANDIDATE = "tidak ada kandidat seri"
GRID_WEIGHTING = "similarity"


def rank_runs(runs: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Urutkan run: F1-macro, lalu F1 judi, lalu training time lebih kecil.

    Untuk RM-c yang tidak melatih apa pun, training time diganti biaya head,
    lalu k dan alpha lebih kecil.

    Args:
        runs: Isi `runs_{scenario}.csv`.
        scenario: Kode skenario.

    Returns:
        Salinan `runs` yang sudah terurut, index di-reset.
    """
    if scenario == "rmc":
        keys = ["val_f1_macro", "val_f1_judi", "head_train_time_s", "k", "alpha"]
    else:
        keys = ["val_f1_macro", "val_f1_judi", "train_time_s"]
    ascending = [False, False] + [True] * (len(keys) - 2)
    return runs.sort_values(keys, ascending=ascending, kind="mergesort").reset_index(drop=True)


def config_of(row: pd.Series, scenario: str) -> dict[str, object]:
    """Hyperparameter satu baris riwayat menurut field model konfigurasi skenario."""
    fields = [name for name in CONFIG_MODELS[scenario].model_fields if name in row.index]
    return {name: _plain(row[name]) for name in fields}


def select_second(
    runs: pd.DataFrame,
    scenario: str,
    first_run_id: int,
    tie_threshold_pp: float | None = None,
) -> tuple[pd.Series | None, list[int]]:
    """Pilih kandidat #2 untuk satu skenario.

    Args:
        runs: Isi `runs_{scenario}.csv`.
        scenario: Kode skenario.
        first_run_id: `run_id` kandidat #1 (juara di `best.json`).
        tie_threshold_pp: Ambang seri; `None` memakai `settings.tie_threshold_pp`.

    Returns:
        Tuple (baris kandidat #2 atau None bila tidak ada yang seri, daftar
        `run_id` yang dikeluarkan karena varian anggaran epoch RM-b).

    Raises:
        ValueError: Kalau `first_run_id` tidak ada di `runs`.
    """
    threshold = settings.tie_threshold_pp if tie_threshold_pp is None else tie_threshold_pp
    matches = runs[runs["run_id"] == first_run_id]
    if matches.empty:
        raise ValueError(f"kandidat #1 run #{first_run_id} tidak ada di riwayat {scenario}")
    first = matches.iloc[0]
    first_config = config_of(first, scenario)

    pool = runs[runs["run_id"] != first_run_id]
    if scenario == "rmc":
        pool = pool[pool["rmb_run_id"] == first["rmb_run_id"]]
    pool = pool[
        (first["val_f1_macro"] - pool["val_f1_macro"]).abs() * PP <= threshold + FLOAT_TOLERANCE
    ]
    pool = pool[[config_of(row, scenario) != first_config for _, row in pool.iterrows()]]

    excluded: list[int] = []
    if scenario == "rmb" and not pool.empty:
        is_variant = pool.apply(lambda row: _same_epoch_variant(row, first), axis=1)
        excluded = [int(run_id) for run_id in pool.loc[is_variant, "run_id"]]
        pool = pool[~is_variant]

    if pool.empty:
        return None, excluded
    return rank_runs(pool, scenario).iloc[0], excluded


def _same_epoch_variant(row: pd.Series, first: pd.Series) -> bool:
    fields = [name for name in CONFIG_MODELS["rmb"].model_fields if name not in ("epochs",)]
    same_except_epochs = all(_plain(row[name]) == _plain(first[name]) for name in fields)
    return bool(same_except_epochs and int(row["best_epoch"]) == int(first["best_epoch"]))


def candidate_entry(row: pd.Series, scenario: str) -> dict[str, object]:
    """Ringkasan satu kandidat untuk `candidates.json`."""
    entry: dict[str, object] = {
        "run_id": int(row["run_id"]),
        "config": config_of(row, scenario),
        "val_f1_macro": float(row["val_f1_macro"]),
        "val_f1_judi": float(row["val_f1_judi"]),
    }
    if "best_epoch" in row.index and pd.notna(row["best_epoch"]):
        entry["best_epoch"] = int(row["best_epoch"])
    return entry


def build_candidates(
    runs_by_scenario: dict[str, pd.DataFrame],
    first_run_ids: dict[str, int],
) -> dict[str, object]:
    """Susun isi `candidates.json` (tanpa hash) untuk seluruh skenario.

    Args:
        runs_by_scenario: Riwayat run tiap skenario.
        first_run_ids: `run_id` kandidat #1 tiap skenario.

    Returns:
        Dict berisi cap waktu, aturan, dan kandidat #1 serta #2 per skenario.
    """
    scenarios: dict[str, object] = {}
    for scenario, runs in runs_by_scenario.items():
        first_id = int(first_run_ids[scenario])
        second, excluded = select_second(runs, scenario, first_id)
        first = runs[runs["run_id"] == first_id].iloc[0]
        entry: dict[str, object] = {
            "first": candidate_entry(first, scenario),
            "second": candidate_entry(second, scenario) if second is not None else None,
            "excluded_epoch_variants": excluded,
        }
        if second is None:
            entry["note"] = NO_TIED_CANDIDATE
        else:
            entry["differs_in"] = {
                name: [entry["first"]["config"][name], value]
                for name, value in entry["second"]["config"].items()
                if entry["first"]["config"].get(name) != value
            }
        scenarios[scenario] = entry

    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tie_threshold_pp": settings.tie_threshold_pp,
        "rule": (
            "kandidat #2 = peringkat tertinggi berikutnya yang seri dengan #1 "
            "(selisih F1-macro validation <= ambang seri); RM-b: varian anggaran epoch "
            "dengan epoch terbaik sama dikeluarkan; RM-a: aturan itu tidak berlaku karena "
            "jadwal linear-warmup bergantung pada total langkah; pemecah seri F1 judi lalu "
            "training time; RM-c: dicari pada head yang sama dengan #1"
        ),
        "scenarios": scenarios,
    }


def content_hash(body: dict[str, object]) -> str:
    """SHA-256 isi kanonik `candidates.json` di luar field hash itu sendiri."""
    canonical = json.dumps(
        {key: value for key, value in body.items() if key != "content_sha256"},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_candidates(path: str | Path, body: dict[str, object]) -> dict[str, object]:
    """Tulis `candidates.json` beserta hash isinya.

    Raises:
        FileExistsError: Kalau berkas sudah ada; kandidat yang sudah ditetapkan
            tidak boleh diganti diam-diam. Hapus berkasnya secara sadar bila
            memang harus ditetapkan ulang sebelum test dibuka.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} sudah ada; kandidat sudah ditetapkan dan tidak ditimpa"
        )
    stamped = {**body, "content_sha256": content_hash(body)}
    write_json(path, stamped)
    return stamped


def load_candidates(path: str | Path) -> dict[str, object]:
    """Baca `candidates.json` dan pastikan isinya tidak berubah sejak ditulis.

    Raises:
        FileNotFoundError: Kalau berkas belum ada.
        ValueError: Kalau hash isi tidak cocok.
    """
    body = read_json(Path(path), default=None)
    if not body:
        raise FileNotFoundError(f"{path} belum ada; tetapkan kandidat di akhir 03c")
    if content_hash(body) != body.get("content_sha256"):
        raise ValueError(f"hash {path} tidak cocok: isi kandidat berubah setelah ditetapkan")
    return body


def _plain(value: object) -> object:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer() and abs(value) >= 1:
        return int(value)
    return value


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


__all__ = [
    "NO_TIED_CANDIDATE",
    "build_candidates",
    "content_hash",
    "load_candidates",
    "rank_runs",
    "select_second",
    "write_candidates",
    "head_group",
    "rac_per_head",
    "rank_rmc_runs",
    "rmc_grid",
    "shared_config",
]
