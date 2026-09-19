"""Eksplorasi RM-c: seluruh head RM-b x seluruh rumus fusi x (alpha, k).

Kampanye RM-c standar menguji fusi linear di atas SATU head, yaitu juara RM-b.
Modul ini menjelajah lebih luas: setiap head yang pernah dilatih di `runs_rmb.csv`
dipasangkan dengan setiap rumus fusi di `FusionFormulaComparator`, supaya klaim
"RAC memperbaiki encoder beku" tidak bertumpu pada satu titik operasi.

Semua evaluasi memakai split validation, indeks FAISS hanya dari train, dan tidak
ada pelatihan: head dimuat dari checkpoint yang disimpan tahap RM-b. Split test
tidak pernah dibuka di sini.

Karena jumlah kandidat jauh lebih banyak daripada grid standar, pemenangnya lebih
rawan bias seleksi. Karena itu penantang baru hanya menggantikan juara standar
bila selisihnya melampaui ambang seri DAN lolos bootstrap berpasangan
(`challenger_wins`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import settings
from src.models.schemas import RMBConfig
from src.services.features import FeatureSet
from src.services.fusion_ablation import FusionFormulaComparator, FusionFormulaConfig
from src.services.rac import NeighborCache
from src.utils.io import read_csv
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

METRIC_PRECISION = 6
PP = 100.0

# Selisih F1-macro antara head yang dimuat dan angka di runs_rmb.csv yang masih
# dianggap head yang sama. Head dari checkpoint sesi yang sama seharusnya
# identik; selisih di atas ini berarti checkpoint tidak berasal dari run itu.
REPRODUCTION_TOLERANCE_PP = 0.05

CONFIG_KEYS = tuple(RMBConfig.model_fields)

# Urutan kesederhanaan rumus saat seri: fusi linear produksi didahulukan, lalu
# sisanya menurut nomor.
FORMULA_PREFERENCE = {"linear": 0, "rumus1": 1, "rumus2": 2, "rumus3": 3, "rumus4": 4}

# Pengganti alpha kosong (rumus tanpa alpha) agar groupby tidak membuang barisnya.
ALPHA_MISSING_KEY = -1.0

DEFAULT_BOOTSTRAPS = 2000
DEFAULT_BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL_PCT = 95.0


def load_exploration_grid(path: str | Path) -> list[FusionFormulaConfig]:
    """Muat grid konfigurasi fusi dari CSV rancangan.

    Args:
        path: Lokasi CSV berkolom `formula`, `alpha`, `k`, `weighting`. `alpha`
            boleh kosong untuk rumus tanpa alpha.

    Returns:
        Daftar konfigurasi sesuai urutan baris.

    Raises:
        ValueError: Kalau berkas kosong atau kolom wajib tidak lengkap.
    """
    frame = read_csv(path)
    if frame.empty:
        raise ValueError(f"grid eksplorasi kosong atau tidak ditemukan: {path}")
    missing = [column for column in ("formula", "k") if column not in frame.columns]
    if missing:
        raise ValueError(f"kolom grid tidak lengkap, hilang: {missing}")

    configs = []
    for row in frame.to_dict("records"):
        alpha = row.get("alpha")
        configs.append(
            FusionFormulaConfig(
                formula=row["formula"],
                alpha=None if alpha is None or pd.isna(alpha) else float(alpha),
                k=int(row["k"]),
                weighting=row.get("weighting") or "similarity",
            )
        )
    return configs


class RMCExplorer:
    """Jalankan grid rumus fusi di atas tiap head RM-b.

    Args:
        features: Embedding beku ketiga split, dari encoder yang sama dengan yang
            dipakai melatih head-head itu.
        device: Device komputasi.
        candidates: Konfigurasi fusi yang diuji di tiap head.

    Raises:
        ValueError: Kalau `candidates` kosong.
    """

    def __init__(
        self,
        features: FeatureSet,
        device: torch.device,
        candidates: list[FusionFormulaConfig],
    ) -> None:
        if not candidates:
            raise ValueError("candidates kosong; tidak ada konfigurasi fusi yang diuji")

        self.features = features
        self.device = device
        self.candidates = candidates

    @staticmethod
    def _head_columns(record: dict[str, object]) -> dict[str, object]:
        return {key: record[key] for key in CONFIG_KEYS if key in record}

    def _explore_head(
        self,
        record: dict[str, object],
        head: nn.Module,
        retrieval: NeighborCache,
    ) -> list[dict[str, object]]:
        run_id = int(record["run_id"])
        comparator = FusionFormulaComparator(
            self.features, head, self.device, k=retrieval.max_k, retrieval=retrieval
        )

        baseline, _ = comparator.evaluate(FusionFormulaConfig("linear", alpha=0.0, k=1))
        logged_f1 = float(record["val_f1_macro"])
        drift_pp = abs(baseline["f1_macro"] - logged_f1) * PP
        if drift_pp > REPRODUCTION_TOLERANCE_PP:
            logger.warning(
                "Head RM-b run #%d tidak cocok dengan runs_rmb.csv: %.4f vs %.4f (%.3f pp)",
                run_id, baseline["f1_macro"], logged_f1, drift_pp,
            )

        rows: list[dict[str, object]] = []
        for config in self.candidates:
            metrics, extras = comparator.evaluate(config)
            alpha_used = extras["alpha_used"]
            rows.append(
                {
                    "rmb_run_id": run_id,
                    **self._head_columns(record),
                    "head_params": int(record["trainable_params"]),
                    "head_train_time_s": float(record["train_time_s"]),
                    "formula": config.formula,
                    "alpha": config.alpha,
                    "alpha_mean": (
                        float(np.mean(alpha_used)) if alpha_used is not None else np.nan
                    ),
                    "k": config.k,
                    "weighting": config.weighting,
                    "val_f1_macro": round(metrics["f1_macro"], METRIC_PRECISION),
                    "val_f1_judi": round(metrics["f1_class1"], METRIC_PRECISION),
                    "val_acc": round(metrics["accuracy"], METRIC_PRECISION),
                    "val_f1_rmb_logged": round(logged_f1, METRIC_PRECISION),
                    "val_f1_rmb": round(baseline["f1_macro"], METRIC_PRECISION),
                    "val_f1_judi_rmb": round(baseline["f1_class1"], METRIC_PRECISION),
                    "gain_pp": round((metrics["f1_macro"] - baseline["f1_macro"]) * PP, 4),
                    "gain_judi_pp": round(
                        (metrics["f1_class1"] - baseline["f1_class1"]) * PP, 4
                    ),
                }
            )
        return rows

    def run(
        self,
        rmb_runs: pd.DataFrame,
        heads: dict[int, nn.Module],
    ) -> pd.DataFrame:
        """Uji seluruh konfigurasi fusi di atas tiap head.

        Args:
            rmb_runs: Isi `runs_rmb.csv` (atau subset-nya): kolom `run_id`,
                `val_f1_macro`, `trainable_params`, `train_time_s`, dan field
                `RMBConfig`.
            heads: Peta `run_id` ke head terlatih; harus memuat setiap `run_id`.

        Returns:
            DataFrame panjang, satu baris per pasangan (head, konfigurasi fusi).

        Raises:
            ValueError: Kalau `rmb_runs` kosong atau ada head yang tidak tersedia.
        """
        if rmb_runs.empty:
            raise ValueError("rmb_runs kosong; tidak ada head yang dieksplorasi")

        records = rmb_runs.to_dict("records")
        unavailable = [int(r["run_id"]) for r in records if int(r["run_id"]) not in heads]
        if unavailable:
            raise ValueError(f"head RM-b run {unavailable} tidak tersedia")

        train_emb, train_lab = self.features["train"]
        retrieval = NeighborCache(
            train_emb, train_lab, max_k=max(config.k for config in self.candidates)
        )

        rows: list[dict[str, object]] = []
        for sequence, record in enumerate(records, start=1):
            run_id = int(record["run_id"])
            logger.info("Head %d/%d: RM-b run #%d", sequence, len(records), run_id)
            rows.extend(self._explore_head(record, heads[run_id], retrieval))

        return pd.DataFrame(rows)


def select_best(
    frame: pd.DataFrame,
    tie_threshold_pp: float | None = None,
) -> pd.Series:
    """Pilih satu konfigurasi terbaik dari sekelompok baris hasil eksplorasi.

    Aturan kampanye: F1-macro validation yang tertinggi, dan konfigurasi dalam
    `tie_threshold_pp` dari yang tertinggi dianggap seri. Pada seri, yang lebih
    murah menang: head dengan parameter lebih sedikit, lalu waktu latih lebih
    singkat (dibulatkan ke detik penuh agar derau pengukuran tidak menentukan),
    lalu fusi `linear`, k lebih kecil, alpha lebih kecil, dan terakhir F1-macro
    lebih tinggi.

    Args:
        frame: Baris hasil eksplorasi; tidak boleh kosong.
        tie_threshold_pp: Ambang seri; `None` memakai `settings.tie_threshold_pp`.

    Returns:
        Baris terpilih.
    """
    threshold = settings.tie_threshold_pp if tie_threshold_pp is None else tie_threshold_pp
    top = frame["val_f1_macro"].max()
    tied = frame[(top - frame["val_f1_macro"]) * PP <= threshold + 1e-9].copy()

    tied["_time"] = tied["head_train_time_s"].round(0)
    tied["_preference"] = tied["formula"].map(FORMULA_PREFERENCE)
    tied["_alpha"] = tied["alpha"].fillna(0.0)

    ordered = tied.sort_values(
        ["head_params", "_time", "_preference", "k", "_alpha", "val_f1_macro"],
        ascending=[True, True, True, True, True, False],
    )
    return ordered.iloc[0].drop(["_time", "_preference", "_alpha"])


def summarize_per_head(sweep: pd.DataFrame) -> pd.DataFrame:
    """Ringkas eksplorasi menjadi satu baris per head RM-b.

    Args:
        sweep: Keluaran `RMCExplorer.run`.

    Returns:
        DataFrame per head: biaya head, baseline head sendiri, konfigurasi fusi
        terbaik untuk head itu (`best_*`), kenaikannya, `rac_helps` (True bila
        kenaikan melampaui ambang seri), dan `on_pareto` (True bila head itu tidak
        didominasi head lain pada F1 terbaik melawan parameter dan waktu latih).
    """
    rows: list[dict[str, object]] = []
    for run_id, group in sweep.groupby("rmb_run_id", sort=True):
        best = select_best(group)
        rows.append(
            {
                "rmb_run_id": run_id,
                **{key: best[key] for key in CONFIG_KEYS if key in best.index and key != "seed"},
                "head_params": best["head_params"],
                "head_train_time_s": best["head_train_time_s"],
                "val_f1_rmb": best["val_f1_rmb"],
                "best_formula": best["formula"],
                "best_alpha": best["alpha"],
                "best_k": best["k"],
                "val_f1_rac_best": best["val_f1_macro"],
                "gain_best_pp": best["gain_pp"],
                "gain_judi_best_pp": best["gain_judi_pp"],
                "rac_helps": bool(best["gain_pp"] > settings.tie_threshold_pp),
            }
        )

    per_head = pd.DataFrame(rows)
    # Waktu dibulatkan ke detik penuh agar derau pengukuran tidak menentukan dominasi.
    costs = per_head.assign(_time=per_head["head_train_time_s"].round(0))
    per_head["on_pareto"] = pareto_front(
        costs, maximize=["val_f1_rac_best"], minimize=["head_params", "_time"]
    )
    return per_head


def summarize_per_formula(sweep: pd.DataFrame) -> pd.DataFrame:
    """Ringkas eksplorasi menjadi satu baris per rumus fusi.

    Untuk tiap rumus dilaporkan konfigurasi terbaik di satu head, dan konfigurasi
    SERAGAM terbaik, yaitu (alpha, k) yang rata-rata F1-macro-nya tertinggi lintas
    seluruh head. Angka seragam menjawab apakah rumus itu membantu tanpa disetel
    per head.

    Args:
        sweep: Keluaran `RMCExplorer.run`.

    Returns:
        DataFrame per rumus.
    """
    rows: list[dict[str, object]] = []
    for formula, group in sweep.groupby("formula", sort=False):
        best = select_best(group)

        keyed = group.assign(_alpha=group["alpha"].fillna(ALPHA_MISSING_KEY))
        averaged = (
            keyed.groupby(["_alpha", "k", "weighting"], as_index=False)[
                ["val_f1_macro", "gain_pp"]
            ]
            .mean()
            .sort_values(["val_f1_macro", "_alpha", "k"], ascending=[False, True, True])
        )
        shared = averaged.iloc[0]
        at_shared = keyed[
            (keyed["_alpha"] == shared["_alpha"])
            & (keyed["k"] == shared["k"])
            & (keyed["weighting"] == shared["weighting"])
        ]
        rows.append(
            {
                "formula": formula,
                "best_rmb_run_id": best["rmb_run_id"],
                "best_alpha": best["alpha"],
                "best_k": best["k"],
                "best_val_f1_macro": best["val_f1_macro"],
                "best_gain_pp": best["gain_pp"],
                "shared_alpha": (
                    np.nan if shared["_alpha"] == ALPHA_MISSING_KEY else shared["_alpha"]
                ),
                "shared_k": shared["k"],
                "shared_mean_val_f1_macro": shared["val_f1_macro"],
                "shared_mean_gain_pp": shared["gain_pp"],
                "shared_heads_helped": int((at_shared["gain_pp"] > settings.tie_threshold_pp).sum()),
                "n_heads": int(at_shared["rmb_run_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def pareto_front(
    frame: pd.DataFrame,
    maximize: list[str],
    minimize: list[str],
) -> pd.Series:
    """Tandai baris yang tidak didominasi baris lain.

    Baris A mendominasi B bila A tidak lebih buruk di semua kriteria dan lebih
    baik di sedikitnya satu.

    Args:
        frame: Tabel kandidat.
        maximize: Kolom yang makin besar makin baik.
        minimize: Kolom yang makin kecil makin baik.

    Returns:
        Series boolean berindeks sama dengan `frame`; True bila tidak didominasi.
    """
    scores = pd.concat(
        [frame[maximize].astype(float), -frame[minimize].astype(float)], axis=1
    ).to_numpy()

    on_front = np.ones(len(scores), dtype=bool)
    for i, row in enumerate(scores):
        not_worse = (scores >= row).all(axis=1)
        strictly_better = (scores > row).any(axis=1)
        if (not_worse & strictly_better).any():
            on_front[i] = False
    return pd.Series(on_front, index=frame.index)


def _macro_f1_per_resample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resamples: np.ndarray,
) -> np.ndarray:
    """F1-macro dua kelas untuk tiap resample, dihitung serentak."""
    truth = y_true[resamples]
    predicted = y_pred[resamples]

    f1_scores = []
    for label in (0, 1):
        true_pos = ((truth == label) & (predicted == label)).sum(axis=1)
        false_pos = ((truth != label) & (predicted == label)).sum(axis=1)
        false_neg = ((truth == label) & (predicted != label)).sum(axis=1)
        denominator = 2 * true_pos + false_pos + false_neg
        f1_scores.append(np.where(denominator > 0, 2 * true_pos / np.maximum(denominator, 1), 0.0))
    return np.mean(f1_scores, axis=0)


def paired_bootstrap(
    y_true: np.ndarray,
    challenger_pred: np.ndarray,
    incumbent_pred: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Selisih F1-macro penantang terhadap petahana beserta interval bootstrap.

    Kedua prediksi dievaluasi pada resample sampel yang SAMA (berpasangan), jadi
    variasi karena sampel yang sulit saling meniadakan. Hanya untuk klasifikasi
    biner, sesuai proyek ini.

    Args:
        y_true: Label sebenarnya (N,).
        challenger_pred: Prediksi penantang (N,).
        incumbent_pred: Prediksi petahana (N,).
        n_boot: Jumlah resample.
        seed: Seed generator acak.

    Returns:
        Tuple (selisih pada data asli, batas bawah, batas atas), semuanya dalam
        poin persentase; interval berupa persentil 95% dari selisih resample.

    Raises:
        ValueError: Kalau panjang ketiga array tidak sama.
    """
    y_true = np.asarray(y_true)
    challenger_pred = np.asarray(challenger_pred)
    incumbent_pred = np.asarray(incumbent_pred)
    if not len(y_true) == len(challenger_pred) == len(incumbent_pred):
        raise ValueError("y_true dan kedua prediksi harus sepanjang sama")

    everything = np.arange(len(y_true))[None, :]
    observed = (
        _macro_f1_per_resample(y_true, challenger_pred, everything)[0]
        - _macro_f1_per_resample(y_true, incumbent_pred, everything)[0]
    )

    resamples = np.random.default_rng(seed).integers(0, len(y_true), size=(n_boot, len(y_true)))
    deltas = _macro_f1_per_resample(
        y_true, challenger_pred, resamples
    ) - _macro_f1_per_resample(y_true, incumbent_pred, resamples)

    tail = (100.0 - CONFIDENCE_LEVEL_PCT) / 2.0
    low, high = np.percentile(deltas, [tail, 100.0 - tail])
    return float(observed * PP), float(low * PP), float(high * PP)


def challenger_wins(
    delta_pp: float,
    ci_low_pp: float,
    tie_threshold_pp: float | None = None,
) -> bool:
    """Apakah penantang layak menggantikan juara standar.

    Syaratnya dua: selisih melampaui ambang seri, dan batas bawah interval
    bootstrap berada di atas nol (selisih tidak sekadar derau sampel).

    Args:
        delta_pp: Selisih F1-macro penantang terhadap petahana, poin persentase.
        ci_low_pp: Batas bawah interval bootstrap, poin persentase.
        tie_threshold_pp: Ambang seri; `None` memakai `settings.tie_threshold_pp`.

    Returns:
        True bila penantang menggantikan petahana.
    """
    threshold = settings.tie_threshold_pp if tie_threshold_pp is None else tie_threshold_pp
    return bool(delta_pp > threshold and ci_low_pp > 0.0)
