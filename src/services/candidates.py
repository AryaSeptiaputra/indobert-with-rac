"""Kandidat #1 dan #2 per skenario untuk uji sensitivitas konfigurasi terpilih.

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


__all__ = [
    "NO_TIED_CANDIDATE",
    "build_candidates",
    "content_hash",
    "load_candidates",
    "rank_runs",
    "select_second",
    "write_candidates",
]
