"""Data mentah untuk Tabel 4.1-4.19 dan L.1, diambil langsung dari berkas keluaran running.

Modul ini TIDAK membuat tabel siap tempel: tabel disusun manual saat penulisan.
Yang disiapkan adalah nilai mentah presisi penuh, satu sheet per nomor tabel di
`artifacts/data_tabel.xlsx`, ditambah sheet `keterangan` (judul, sumber berkas,
catatan, status). Tidak ada angka yang diketik manual.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import settings
from src.services.evaluation import ClassificationEvaluator
from src.services.selection import head_group, load_candidates, rac_per_head, rank_rmc_runs, rank_runs, rmc_grid
from src.utils.io import read_csv, read_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

PP = 100.0
TOP_N = 3
SCENARIO_LABELS = {"rma": "RM-a", "rmb": "RM-b", "rmc": "RM-c"}
MAX_F1_GAP_PP = 3.0
MIN_PARAM_REDUCTION_PCT = 90.0
MIN_TIME_REDUCTION_PCT = 50.0
MIN_CRITERIA = 2

RMA_STAGES = {
    "": "Tahap 1: grid lr x epoch x batch",
    "rma_tahap1_grid": "Tahap 1: grid lr x epoch x batch",
    "rma_tahap2_coordinate": "Tahap 2: coordinate descent",
}
RMB_STAGES = {
    "rmb_tuning_grid": "Tahap 1: arsitektur head",
    "rmb_tuning_grid_stage1b": "Tahap 1B: perluasan dimensi tersembunyi",
    "rmb_tuning_grid_stage2": "Tahap 2: grid lr x epoch",
    "rmb_tuning_grid_stage3": "Tahap 3: coordinate descent",
}
RMC_STAGES = {
    "rmc_grid_seluruh_head": "Tahap 1: seluruh head RM-b x alpha x k",
    "rmc_tahap2_weighting": "Tahap 2: verifikasi skema pembobotan",
}
CAPACITY_BATCHES = ("rmb_tuning_grid", "rmb_tuning_grid_stage1b")


@dataclass
class TableData:
    """Data mentah satu tabel.

    Attributes:
        number: Nomor tabel, misalnya "4.1" atau "L.1".
        title: Judul tabel.
        frame: Nilai mentah presisi penuh.
        sources: Berkas keluaran running yang dibaca.
        notes: Konteks yang dibutuhkan saat menyusun tabel.
    """

    number: str
    title: str
    frame: pd.DataFrame
    sources: list[str]
    notes: list[str] = field(default_factory=list)


def head_label(head_arch: str, hidden_dim: object) -> str:
    """`linear` atau `MLP <hidden_dim>`."""
    return "linear" if head_arch == "linear" else f"MLP {int(hidden_dim)}"


def reduction_pct(value: float | None, baseline: float | None) -> float | None:
    """Reduksi terhadap baseline dalam persen; None bila salah satunya tidak ada."""
    if value is None or baseline is None or pd.isna(value) or pd.isna(baseline) or not baseline:
        return None
    return (1.0 - float(value) / float(baseline)) * PP


class ThesisDataExporter:
    """Kumpulkan data mentah seluruh tabel untuk satu folder keluaran kampanye.

    Args:
        out_dir: Folder keluaran kampanye; `None` memakai `settings.default_out_dir`.
        metadata_path: `metadata.json` split; `None` memakai `settings.metadata_path`.
        preprocessing_stats_path: Statistik dari notebook 02; `None` memakai
            `outputs/preprocessing/preprocessing_stats.json`.
    """

    def __init__(
        self,
        out_dir: str | Path | None = None,
        metadata_path: str | Path | None = None,
        preprocessing_stats_path: str | Path | None = None,
    ) -> None:
        self.out_dir = settings.resolve_out_dir(out_dir)
        self.metadata_path = Path(metadata_path) if metadata_path else settings.metadata_path
        self.preprocessing_stats_path = (
            Path(preprocessing_stats_path)
            if preprocessing_stats_path
            else settings.output_dir / "preprocessing" / "preprocessing_stats.json"
        )
        self.export_path = self.out_dir / "artifacts" / "data_tabel.xlsx"
        self.evaluator = ClassificationEvaluator()

    def collectors(self) -> dict[str, Callable[[], TableData]]:
        """Peta nomor tabel ke fungsi pengumpul datanya, urut sesuai Bab 4."""
        return {
            "4.1": self.table_4_1, "4.2": self.table_4_2, "4.3": self.table_4_3,
            "4.4": self.table_4_4, "4.5": self.table_4_5, "4.6": self.table_4_6,
            "4.7": self.table_4_7, "4.8": self.table_4_8, "4.9": self.table_4_9,
            "4.10": self.table_4_10, "4.11": self.table_4_11, "4.12": self.table_4_12,
            "4.13": self.table_4_13, "4.14": self.table_4_14, "4.15": self.table_4_15,
            "4.16": self.table_4_16, "4.17": self.table_4_17, "4.18": self.table_4_18,
            "4.19": self.table_4_19, "L.1": self.table_l_1,
        }

    def collect_all(self, skip_missing: bool = True) -> dict[str, TableData | str]:
        """Kumpulkan data seluruh tabel.

        Args:
            skip_missing: Lewati tabel yang berkas keluarannya belum ada (misalnya
                data test sebelum 05 dijalankan) alih-alih berhenti.

        Returns:
            Peta nomor tabel ke `TableData`, atau alasan tabel itu dilewati.

        Raises:
            FileNotFoundError: Kalau `skip_missing=False` dan ada berkas yang hilang.
        """
        results: dict[str, TableData | str] = {}
        for number, collect in self.collectors().items():
            try:
                results[number] = collect()
            except FileNotFoundError as exc:
                if not skip_missing:
                    raise
                logger.warning("Data tabel %s dilewati: %s", number, exc)
                results[number] = f"dilewati: {exc}"
        return results

    def export(self, results: dict[str, TableData | str]) -> Path:
        """Tulis `artifacts/data_tabel.xlsx`: sheet `keterangan` lalu satu sheet per tabel.

        Args:
            results: Keluaran `collect_all`.

        Returns:
            Path berkas yang ditulis.
        """
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        index = [
            {
                "nomor": number,
                "judul": item.title if isinstance(item, TableData) else "",
                "status": "tersedia" if isinstance(item, TableData) else item,
                "sumber_berkas": "; ".join(item.sources) if isinstance(item, TableData) else "",
                "catatan": " | ".join(item.notes) if isinstance(item, TableData) else "",
                "dibangkitkan_pada": generated_at,
            }
            for number, item in results.items()
        ]
        with pd.ExcelWriter(self.export_path, engine="openpyxl") as writer:
            pd.DataFrame(index).to_excel(writer, sheet_name="keterangan", index=False)
            for number, item in results.items():
                if isinstance(item, TableData):
                    item.frame.to_excel(writer, sheet_name=number, index=False)
        logger.info("Data tabel ditulis: %s", self.export_path)
        return self.export_path

    # ------------------------------------------------------------------
    # 4.1 - 4.4: lingkungan dan data
    # ------------------------------------------------------------------

    def table_4_1(self) -> TableData:
        """Spesifikasi lingkungan eksperimen dan status sesi."""
        hardware = self._json(self.out_dir / "hardware.json", "hardware.json")
        keys = ("gpu", "vram_total_mb", "driver", "cuda_version", "cpu", "cpu_logical_cores",
                "ram_total_gb", "os", "python", "torch", "transformers", "faiss", "seed")
        rows = [{"komponen": key, "nilai": hardware.get(key)} for key in keys]
        sessions = hardware.get("tuning_sessions", [])
        for i, session in enumerate(sessions, start=1):
            rows.append({"komponen": f"sesi_tuning_{i}_recorded_at", "nilai": session["recorded_at"]})
            rows.append({"komponen": f"sesi_tuning_{i}_boot_time", "nilai": session["boot_time"]})
        final = hardware.get("final_session")
        if final:
            rows += [
                {"komponen": "sesi_final_recorded_at", "nilai": final["recorded_at"]},
                {"komponen": "sesi_final_boot_time", "nilai": final["boot_time"]},
                {"komponen": "sesi_final_same_boot_as_tuning", "nilai": final.get("same_boot_as_tuning")},
            ]
        notes = [
            "Status sesi: satu sesi mesin bila hanya ada satu sesi tuning dan sesi_final_same_boot_as_tuning=True; "
            "selain itu hardware sama tetapi sesi mesin berbeda."
        ]
        return TableData("4.1", "Spesifikasi lingkungan eksperimen", pd.DataFrame(rows), ["hardware.json"], notes)

    def table_4_2(self) -> TableData:
        """Jumlah baris per tahap pra-pemrosesan dan baris terbuang per kelas di guard."""
        counts = self._json(self.metadata_path, "metadata.json")["counts"]
        stats = self._json(self.preprocessing_stats_path, "preprocessing_stats.json")
        stages = [("mentah", counts["raw"]), ("buang baris kosong", counts["after_missing"]),
                  ("deduplikasi NFKC", counts["after_dedup"]), ("guard anti-leakage", counts["final_total"])]
        rows, previous = [], None
        for name, count in stages:
            change = None if previous is None else count - previous
            rows.append({"tahap": name, "jumlah_baris": count, "perubahan_baris": change,
                         "perubahan_pct": None if previous is None else change / previous * PP})
            previous = count
        removed = stats.get("leakage_removed_by_class", {})
        rows.append({"tahap": "guard: terbuang kelas judi", "jumlah_baris": removed.get("judi", 0)})
        rows.append({"tahap": "guard: terbuang kelas non-judi", "jumlah_baris": removed.get("non_judi", 0)})
        return TableData("4.2", "Reduksi jumlah baris per tahap pra-pemrosesan", pd.DataFrame(rows),
                         ["data/processed/metadata.json", "outputs/preprocessing/preprocessing_stats.json"])

    def table_4_3(self) -> TableData:
        """Persentase baris ber-[UNK] sebelum dan sesudah NFKC, per kelas."""
        nfkc = self._json(self.preprocessing_stats_path, "preprocessing_stats.json")["nfkc_unk"]
        rates = nfkc["rates_pct"]
        rows = [
            {"kondisi": "sebelum_nfkc", "unk_judi_pct": rates["sebelum_nfkc"]["judi"],
             "unk_non_judi_pct": rates["sebelum_nfkc"]["non_judi"]},
            {"kondisi": "sesudah_nfkc", "unk_judi_pct": rates["sesudah_nfkc"]["judi"],
             "unk_non_judi_pct": rates["sesudah_nfkc"]["non_judi"]},
        ]
        notes = [
            f"{nfkc['unit']}; tokenizer pelatihan (memuat [URL], [MENTION], [NUM]); sebelum NFKC = pembersihan "
            f"yang sama tanpa langkah NFKC; dataset final {nfkc['n_rows']['judi']} baris judi, "
            f"{nfkc['n_rows']['non_judi']} baris non-judi."
        ]
        return TableData("4.3", "Dampak normalisasi NFKC", pd.DataFrame(rows),
                         ["outputs/preprocessing/preprocessing_stats.json"], notes)

    def table_4_4(self) -> TableData:
        """Komposisi dataset final, class weight, dan konflik label."""
        metadata = self._json(self.metadata_path, "metadata.json")
        rows = []
        for name in ("train", "val", "test"):
            split = metadata["splits"][name]
            rows.append({"split": name, "jumlah_baris": split["n"], "kelas_0": split["L0"], "kelas_1": split["L1"],
                         "persen_kelas_1": split["L1"] / split["n"] * PP})
        frame = pd.DataFrame(rows)
        total = frame[["jumlah_baris", "kelas_0", "kelas_1"]].sum()
        frame.loc[len(frame)] = {"split": "total", **total.to_dict(),
                                 "persen_kelas_1": total["kelas_1"] / total["jumlah_baris"] * PP}
        weights = metadata["class_weights"]
        frame["class_weight"] = [f"0: {weights['0']}; 1: {weights['1']}" if s == "train" else None for s in frame["split"]]
        conflict = metadata.get("label_conflict", {})
        notes = [f"Class weight dari train. Konflik label: {conflict.get('groups')} kelompok, "
                 f"{conflict.get('rows')} baris, kebijakan {conflict.get('policy')}."]
        return TableData("4.4", "Dataset final", frame, ["data/processed/metadata.json"], notes)

    # ------------------------------------------------------------------
    # 4.5 - 4.7: ruang pencarian yang dieksekusi
    # ------------------------------------------------------------------

    def table_4_5(self) -> TableData:
        """Ruang pencarian RM-a yang benar-benar dieksekusi."""
        params = ["lr", "epochs", "batch", "warmup_ratio", "weight_decay", "micro_batch"]
        return self._search_space("4.5", "Ruang pencarian hyperparameter RM-a", self._runs("rma"),
                                  RMA_STAGES, params, "runs_rma.csv")

    def table_4_6(self) -> TableData:
        """Ruang pencarian RM-b yang benar-benar dieksekusi."""
        runs = self._runs("rmb").copy()
        runs["head"] = [head_label(a, h) for a, h in zip(runs["head_arch"], runs["hidden_dim"], strict=True)]
        params = ["head", "lr", "epochs", "dropout", "weight_decay", "batch"]
        return self._search_space("4.6", "Ruang pencarian hyperparameter RM-b", runs, RMB_STAGES, params, "runs_rmb.csv")

    def table_4_7(self) -> TableData:
        """Ruang pencarian RM-c, jumlah evaluasi, dan indeks FAISS."""
        runs = self._runs("rmc")
        data = self._search_space("4.7", "Ruang pencarian RM-c", runs, RMC_STAGES,
                                  ["rmb_run_id", "alpha", "k", "weighting"], "runs_rmc.csv")
        train_rows = self._json(self.metadata_path, "metadata.json")["splits"]["train"]["n"]
        vectors = sorted(int(v) for v in runs["index_vectors"].dropna().unique())
        index_types = sorted(str(v) for v in runs["index_type"].dropna().unique()) if "index_type" in runs else []
        extra = pd.DataFrame([
            {"tahap": "indeks FAISS", "parameter": "vektor_dalam_indeks", "nilai": "; ".join(map(str, vectors)),
             "jumlah": len(vectors)},
            {"tahap": "indeks FAISS", "parameter": "jumlah_baris_train", "nilai": str(train_rows), "jumlah": 1},
            {"tahap": "indeks FAISS", "parameter": "sama_dengan_train", "nilai": str(vectors == [train_rows]), "jumlah": 1},
            {"tahap": "indeks FAISS", "parameter": "jenis_indeks", "nilai": "; ".join(index_types), "jumlah": len(index_types)},
        ])
        data.frame = pd.concat([data.frame, extra], ignore_index=True)
        data.sources.append("data/processed/metadata.json")
        return data

    def _search_space(self, number, title, runs, stage_labels, params, source) -> TableData:
        runs = runs.sort_values("run_id").copy()
        runs["_stage"] = runs["batch_id"].fillna("").astype(str).map(
            lambda batch: stage_labels.get(batch, batch or "run tunggal")
        )
        rows = []
        for stage in dict.fromkeys(runs["_stage"]):
            subset = runs[runs["_stage"] == stage]
            for param in params:
                values = sorted(subset[param].dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
                if len(values) > 1:
                    rows.append({"tahap": stage, "parameter": param, "nilai": "; ".join(map(str, values)),
                                 "jumlah": len(values)})
            rows.append({"tahap": stage, "parameter": "konfigurasi_unik", "nilai": "",
                         "jumlah": len(subset[params].drop_duplicates())})
            rows.append({"tahap": stage, "parameter": "eksekusi", "nilai": "", "jumlah": len(subset)})
        rows.append({"tahap": "total", "parameter": "konfigurasi_unik", "nilai": "",
                     "jumlah": len(runs[params].drop_duplicates())})
        rows.append({"tahap": "total", "parameter": "eksekusi", "nilai": "", "jumlah": len(runs)})
        for param in params:
            if runs[param].dropna().nunique() == 1:
                rows.append({"tahap": "tetap di seluruh tahap", "parameter": param,
                             "nilai": str(runs[param].dropna().iloc[0]), "jumlah": 1})
        return TableData(number, title, pd.DataFrame(rows), [source])

    # ------------------------------------------------------------------
    # 4.8 - 4.12: hasil tuning pada validation
    # ------------------------------------------------------------------

    def table_4_8(self) -> TableData:
        """Tiga konfigurasi terbaik RM-a pada validation."""
        columns = ["run_id", "lr", "epochs", "batch", "warmup_ratio", "weight_decay", "best_epoch",
                   "val_f1_macro", "val_precision_macro", "val_recall_macro", "val_f1_judi", "train_time_s"]
        return self._top("4.8", "Tiga konfigurasi terbaik RM-a pada validation", "rma", columns)

    def table_4_9(self) -> TableData:
        """Kapasitas classification head terhadap F1 RM-b."""
        runs = self._runs("rmb")
        points = runs[runs["batch_id"].isin(CAPACITY_BATCHES)].sort_values("trainable_params").reset_index(drop=True)
        if points.empty:
            raise FileNotFoundError("runs_rmb.csv belum memuat tahap arsitektur head")
        frame = pd.DataFrame({
            "run_id": points["run_id"],
            "arsitektur_head": [head_label(a, h) for a, h in zip(points["head_arch"], points["hidden_dim"], strict=True)],
            "trainable_params": points["trainable_params"],
            "val_f1_macro": points["val_f1_macro"],
        })
        frame["kenaikan_marjinal_pp_per_100rb_param"] = (
            frame["val_f1_macro"].diff() * PP / frame["trainable_params"].diff() * 1e5
        )
        return TableData("4.9", "Kapasitas classification head terhadap F1 RM-b", frame, ["runs_rmb.csv"],
                         ["Kenaikan marjinal terhadap arsitektur sebelumnya pada urutan trainable params."])

    def table_4_10(self) -> TableData:
        """Tiga konfigurasi terbaik RM-b pada validation."""
        columns = ["run_id", "head_arch", "hidden_dim", "lr", "epochs", "dropout", "weight_decay", "batch",
                   "best_epoch", "val_f1_macro", "val_f1_judi", "train_time_s"]
        data = self._top("4.10", "Tiga konfigurasi terbaik RM-b pada validation", "rmb", columns)
        candidates_path = self.out_dir / "candidates.json"
        if candidates_path.exists():
            excluded = load_candidates(candidates_path)["scenarios"]["rmb"].get("excluded_epoch_variants", [])
            data.notes.append(
                "Varian anggaran epoch dengan epoch terbaik sama dengan juara (bobot identik, learning rate RM-b "
                f"konstan): run {excluded if excluded else 'tidak ada'}."
            )
            data.sources.append("candidates.json")
        return data

    def _top(self, number: str, title: str, scenario: str, columns: list[str]) -> TableData:
        runs = self._runs(scenario)
        champion_id = int(self._best()[scenario]["run_id"])
        champion_f1 = float(runs.loc[runs["run_id"] == champion_id, "val_f1_macro"].iloc[0])
        top = rank_runs(runs, scenario).head(TOP_N)
        frame = top[[c for c in columns if c in top.columns]].copy()
        frame.insert(0, "peringkat", range(1, len(frame) + 1))
        frame["selisih_terhadap_juara_pp"] = (frame["val_f1_macro"] - champion_f1) * PP
        frame["status"] = [
            "juara" if int(run_id) == champion_id
            else ("seri" if abs(delta) <= settings.tie_threshold_pp + 1e-9 else "tidak seri")
            for run_id, delta in zip(frame["run_id"], frame["selisih_terhadap_juara_pp"], strict=True)
        ]
        notes = [f"Peringkat: F1-macro, lalu F1 judi, lalu training time. Seri = selisih <= {settings.tie_threshold_pp} pp dari juara."]
        return TableData(number, title, frame.reset_index(drop=True), [f"runs_{scenario}.csv", "best.json"], notes)

    def table_4_11(self) -> TableData:
        """Tiga konfigurasi terbaik RM-c, default, penantang, final, dan bootstrap."""
        runs = self._runs("rmc")
        decision = self._json(self.out_dir / "rmc_champion_decision.json", "rmc_champion_decision.json")
        baseline = rmc_grid(runs).groupby("rmb_run_id")["f1_no_rac"].first()
        ranked = rank_rmc_runs(runs)

        roles: dict[int, list[str]] = {}
        for rank, run_id in enumerate(ranked["run_id"].head(TOP_N), start=1):
            roles.setdefault(int(run_id), []).append(f"peringkat {rank}")
        roles.setdefault(int(decision["challenger"]["run_id"]), []).append("penantang")
        roles.setdefault(int(decision["default"]["run_id"]), []).append("default (head RM-b resmi)")
        roles.setdefault(int(self._best()["rmc"]["run_id"]), []).append("final")

        rows = []
        for run_id, role in roles.items():
            row = runs[runs["run_id"] == run_id].iloc[0]
            no_rac = float(baseline[int(row["rmb_run_id"])])
            rows.append({"peran": ", ".join(role), "run_id": run_id, "head_rmb_run_id": int(row["rmb_run_id"]),
                         "alpha": row["alpha"], "k": int(row["k"]), "weighting": row["weighting"],
                         "val_f1_macro": row["val_f1_macro"], "val_f1_judi": row["val_f1_judi"],
                         "val_f1_tanpa_rac": no_rac, "selisih_terhadap_tanpa_rac_pp": (row["val_f1_macro"] - no_rac) * PP})
        bootstrap = decision.get("bootstrap") or {}
        notes = [
            decision["rule"],
            f"putusan: {decision['winner']}; head resmi RM-b #{decision['official_rmb_run_id']}; "
            f"head_is_official_rmb={decision['head_is_official_rmb']}",
            (f"bootstrap: observed_delta_pp={bootstrap.get('observed_delta_pp')}, mean_delta_pp={bootstrap.get('mean_delta_pp')}, "
             f"ci95_pp=[{bootstrap.get('ci_low_pp')}, {bootstrap.get('ci_high_pp')}], n_boot={bootstrap.get('n_boot')}, "
             f"seed={bootstrap.get('seed')}, n_samples={bootstrap.get('n_samples')}")
            if bootstrap else "penantang sama dengan default; bootstrap tidak diperlukan",
            decision["cost_note"],
        ]
        return TableData("4.11", "Tiga konfigurasi terbaik RM-c pada validation", pd.DataFrame(rows),
                         ["runs_rmc.csv", "rmc_champion_decision.json", "best.json"], notes)

    def table_4_12(self) -> TableData:
        """Kontribusi RAC per kelompok head (kelompok menurut arsitektur)."""
        summary = self._rac_summary()
        summary["kelompok"] = [head_group(a, h) for a, h in zip(summary["head_arch"], summary["hidden_dim"], strict=True)]
        rows = []
        for group, subset in summary.groupby("kelompok", sort=False):
            rows.append({
                "kelompok_head": group, "jumlah_head": len(subset),
                "f1_tanpa_rac_min": subset["f1_no_rac"].min(), "f1_tanpa_rac_maks": subset["f1_no_rac"].max(),
                "f1_dengan_rac_min": subset["f1_best"].min(), "f1_dengan_rac_maks": subset["f1_best"].max(),
                "median_gain_terbaik_pp": subset["gain_best_pp"].median(),
                "median_gain_konfigurasi_bersama_pp": subset["gain_shared_pp"].median(),
                "median_gain_f1_judi_pp": subset["gain_judi_best_pp"].median(),
            })
        order = {"Linear": 0, "MLP hidden dim < 1024": 1}
        frame = pd.DataFrame(rows).sort_values("kelompok_head", key=lambda s: s.map(lambda g: order.get(g, 2)))
        notes = [
            "Kelompok ditetapkan menurut arsitektur head, bukan F1. Kelompok berisi 1 head: tampilkan nilainya langsung.",
            "F1 tanpa RAC = baris alpha = 0 di grid RM-c (head dimuat dari checkpoint).",
            f"Konfigurasi bersama (rata-rata gain tertinggi di seluruh head): alpha={summary['shared_alpha'].iloc[0]}, "
            f"k={summary['shared_k'].iloc[0]}.",
        ]
        return TableData("4.12", "Kontribusi RAC per kelompok head RM-b", frame.reset_index(drop=True),
                         ["runs_rmc.csv", "runs_rmb.csv"], notes)

    # ------------------------------------------------------------------
    # 4.13 - 4.16: perbandingan final
    # ------------------------------------------------------------------

    def table_4_13(self) -> TableData:
        """Metrik validation konfigurasi final ketiga skenario."""
        rows = {s: self._champion_row(s) for s in SCENARIO_LABELS}
        params = {"rma": rows["rma"]["trainable_params"], "rmb": rows["rmb"]["trainable_params"],
                  "rmc": rows["rmc"]["head_trainable_params"]}
        keys = ["val_f1_macro", "val_acc", "val_precision_macro", "val_recall_macro", "val_f1_judi"]
        frame = pd.DataFrame(
            [{"metrik": "run_id", **{SCENARIO_LABELS[s]: int(r["run_id"]) for s, r in rows.items()}},
             {"metrik": "konfigurasi", **{SCENARIO_LABELS[s]: str(self._best()[s]["config"]) for s in rows}},
             {"metrik": "trainable_params", **{SCENARIO_LABELS[s]: int(v) for s, v in params.items()}}]
            + [{"metrik": key, **{SCENARIO_LABELS[s]: float(r[key]) for s, r in rows.items()}} for key in keys]
            + [{"metrik": "selisih_f1_macro_terhadap_rma_pp",
                **{SCENARIO_LABELS[s]: None if s == "rma" else (float(r["val_f1_macro"]) - float(rows["rma"]["val_f1_macro"])) * PP
                   for s, r in rows.items()}}]
        )
        notes = ["trainable_params RM-c = milik head RM-b yang dipakainya; RM-c sendiri tidak melatih apa pun."]
        return TableData("4.13", "Perbandingan konfigurasi final ketiga skenario pada validation", frame,
                         ["best.json", "runs_rma.csv", "runs_rmb.csv", "runs_rmc.csv"], notes)

    def table_4_14(self) -> TableData:
        """Metrik test dan confusion matrix ketiga skenario."""
        results = self.test_results()
        keys = ["f1_macro", "accuracy", "precision_macro", "recall_macro", "precision_class1", "recall_class1",
                "f1_class1", "tp", "fp", "fn", "tn"]
        frame = pd.DataFrame(
            [{"metrik": key, **{SCENARIO_LABELS[s]: r[key] for s, r in results.items()}} for key in keys]
            + [{"metrik": "selisih_f1_macro_terhadap_rma_pp",
                **{SCENARIO_LABELS[s]: None if s == "rma" else (r["f1_macro"] - results["rma"]["f1_macro"]) * PP
                   for s, r in results.items()}}]
        )
        return TableData("4.14", "Hasil klasifikasi pada test set", frame, ["metrics/final_predictions.csv"],
                         ["class1 = kelas judi; TP/FP/FN/TN dengan kelas judi sebagai positif."])

    def table_4_15(self) -> TableData:
        """F1 validation dan test serta selisihnya."""
        results = self.test_results()
        val = {s: float(self._champion_row(s)["val_f1_macro"]) for s in SCENARIO_LABELS}
        rows = [{
            "skenario": label, "val_f1_macro": val[s], "test_f1_macro": results[s]["f1_macro"],
            "perubahan_val_ke_test_pp": (results[s]["f1_macro"] - val[s]) * PP,
            "selisih_terhadap_rma_val_pp": None if s == "rma" else (val[s] - val["rma"]) * PP,
            "selisih_terhadap_rma_test_pp": None if s == "rma" else (results[s]["f1_macro"] - results["rma"]["f1_macro"]) * PP,
        } for s, label in SCENARIO_LABELS.items()]
        return TableData("4.15", "Selisih validation ke test", pd.DataFrame(rows),
                         ["best.json", "runs_*.csv", "metrics/final_predictions.csv"])

    def table_4_16(self) -> TableData:
        """Kandidat #1 dan #2 per skenario pada validation dan test."""
        candidates = load_candidates(self.out_dir / "candidates.json")
        tested = self._metrics_csv("candidates_test.csv")
        if set(tested["candidates_sha256"].astype(str)) != {candidates["content_sha256"]}:
            raise ValueError("hash kandidat di candidates_test.csv tidak cocok dengan candidates.json")

        rows = []
        for scenario, label in SCENARIO_LABELS.items():
            entry = candidates["scenarios"][scenario]
            second = entry.get("second")
            if not second:
                rows.append({"skenario": label, "perbedaan_kandidat_2": entry.get("note")})
                continue
            test = tested[tested["scenario"] == scenario].set_index("rank")["test_f1_macro"]
            rows.append({
                "skenario": label, "perbedaan_kandidat_2": str(entry["differs_in"]),
                "run_id_1": entry["first"]["run_id"], "run_id_2": second["run_id"],
                "val_f1_1": entry["first"]["val_f1_macro"], "val_f1_2": second["val_f1_macro"],
                "test_f1_1": float(test[1]), "test_f1_2": float(test[2]),
                "sebaran_test_pp": abs(float(test[1]) - float(test[2])) * PP,
            })
        notes = [candidates["rule"],
                 f"ditetapkan {candidates['created_at']}; sha256 {candidates['content_sha256']}; hash di candidates_test.csv cocok."]
        return TableData("4.16", "Sensitivitas terhadap konfigurasi terpilih", pd.DataFrame(rows),
                         ["candidates.json", "metrics/candidates_test.csv"], notes)

    # ------------------------------------------------------------------
    # 4.17 - 4.19: efisiensi dan putusan
    # ------------------------------------------------------------------

    def efficiency(self) -> dict[str, dict[str, object]]:
        """Angka efisiensi per skenario untuk Tabel 4.17-4.19."""
        best = self._best()
        rma = self._champion_row("rma")
        heads = self._runs("rmb").set_index("run_id")
        benchmark = self._metrics_csv("inference_benchmark.csv").set_index("scenario")
        index = self._json(self.out_dir / "metrics" / "index_stats.json", "metrics/index_stats.json")

        def head_costs(run_id: int) -> dict[str, object]:
            row = heads.loc[run_id]
            return {"head_rmb_run_id": run_id, "trainable_params": float(row["trainable_params"]),
                    "train_time_s": float(row["train_time_s"]), "extract_time_s": float(row["extract_time_s"]),
                    "head_train_time_s": float(row["head_train_time_s"]),
                    "train_peak_mem_mb": float(row["train_peak_mem_mb"])}

        values: dict[str, dict[str, object]] = {
            "rma": {"head_rmb_run_id": None, "trainable_params": float(rma["trainable_params"]),
                    "train_time_s": float(rma["train_time_s"]), "extract_time_s": None, "head_train_time_s": None,
                    "train_peak_mem_mb": float(rma["peak_mem_mb"]), "index_build_s": None, "index_size_mb": None},
            "rmb": {**head_costs(int(best["rmb"]["run_id"])), "index_build_s": None, "index_size_mb": None},
            "rmc": {**head_costs(int(best["rmc"]["config"]["rmb_run_id"])),
                    "index_build_s": float(index["build_time_s"]), "index_size_mb": float(index["size_mb"])},
        }
        for scenario, label in SCENARIO_LABELS.items():
            for key in ("infer_peak_gpu_mem_mb", "infer_latency_ms", "latency_warmup_runs", "latency_runs"):
                values[scenario][key] = benchmark.loc[label, key] if key in benchmark.columns else None
        values["_index"] = index
        return values

    def table_4_17(self) -> TableData:
        """Biaya pelatihan dan inferensi ketiga skenario serta reduksinya terhadap RM-a."""
        values = self.efficiency()
        keys = ["head_rmb_run_id", "trainable_params", "train_time_s", "extract_time_s", "head_train_time_s",
                "index_build_s", "train_peak_mem_mb", "infer_peak_gpu_mem_mb", "index_size_mb", "infer_latency_ms",
                "latency_warmup_runs", "latency_runs"]
        rows = []
        for key in keys:
            row = {"metrik": key, **{SCENARIO_LABELS[s]: values[s][key] for s in SCENARIO_LABELS}}
            if key not in ("head_rmb_run_id", "latency_warmup_runs", "latency_runs"):
                row["reduksi_rmb_pct"] = reduction_pct(values["rmb"][key], values["rma"][key])
                row["reduksi_rmc_pct"] = reduction_pct(values["rmc"][key], values["rma"][key])
            rows.append(row)
        index = values["_index"]
        notes = [
            "train_time_s RM-b/RM-c = extract_time_s + head_train_time_s; RM-c memakai head hasil putusan juara "
            "(head_rmb_run_id), index_build_s dicatat terpisah.",
            "train_peak_mem_mb RM-b/RM-c termasuk ekstraksi fitur (maks ekstraksi dan latih head).",
            "Latency = jumlah komponen dari satu jalur yang sama, rata-rata latency_runs pengulangan setelah "
            "latency_warmup_runs pemanasan.",
            f"Indeks {index.get('index_type')}: {index.get('vectors')} vektor, dimensi {index.get('dimension')}, dari split {index.get('source_split')}.",
            "Biaya latih dari sesi tuning, biaya inferensi dari sesi benchmark final, hardware sama (Tabel 4.1). "
            "Reduksi negatif = lebih besar dari RM-a.",
        ]
        return TableData("4.17", "Perbandingan efisiensi komputasi", pd.DataFrame(rows),
                         ["best.json", "runs_rma.csv", "runs_rmb.csv", "metrics/inference_benchmark.csv",
                          "metrics/index_stats.json"], notes)

    def criteria(self) -> dict[str, dict[str, object]]:
        """Kriteria sukses RM-b dan RM-c, termasuk terhadap RM-a termurah."""
        results = self.test_results()
        values = self.efficiency()
        cheapest = self._runs("rma").sort_values(["train_time_s", "run_id"]).iloc[0]
        outcome: dict[str, dict[str, object]] = {"_cheapest_rma": cheapest.to_dict()}
        for scenario in ("rmb", "rmc"):
            delta = (results[scenario]["f1_macro"] - results["rma"]["f1_macro"]) * PP
            params = reduction_pct(values[scenario]["trainable_params"], values["rma"]["trainable_params"])
            time_red = reduction_pct(values[scenario]["train_time_s"], values["rma"]["train_time_s"])
            worst = reduction_pct(values[scenario]["train_time_s"], float(cheapest["train_time_s"]))
            passed = [delta >= -MAX_F1_GAP_PP, params >= MIN_PARAM_REDUCTION_PCT, time_red >= MIN_TIME_REDUCTION_PCT]
            outcome[scenario] = {
                "selisih_f1_test_terhadap_rma_pp": delta,
                "reduksi_trainable_params_pct": params,
                "reduksi_training_time_pct": time_red,
                "reduksi_training_time_terhadap_rma_termurah_pct": worst,
                "jarak_ambang_f1_pp": delta + MAX_F1_GAP_PP,
                "jarak_ambang_params_pp": params - MIN_PARAM_REDUCTION_PCT,
                "jarak_ambang_training_time_pp": time_red - MIN_TIME_REDUCTION_PCT,
                "jarak_ambang_training_time_termurah_pp": worst - MIN_TIME_REDUCTION_PCT,
                "kriteria_terpenuhi": sum(passed),
                "kompetitif": sum(passed) >= MIN_CRITERIA,
            }
        return outcome

    def table_4_18(self) -> TableData:
        """Evaluasi kriteria sukses."""
        outcome = self.criteria()
        cheapest = outcome.pop("_cheapest_rma")
        frame = pd.DataFrame([{"skenario": SCENARIO_LABELS[s], **item} for s, item in outcome.items()])
        notes = [
            f"Ambang: selisih F1 >= -{MAX_F1_GAP_PP} pp, reduksi params >= {MIN_PARAM_REDUCTION_PCT}%, reduksi "
            f"training time >= {MIN_TIME_REDUCTION_PCT}% terhadap juara RM-a; kompetitif bila >= {MIN_CRITERIA} dari 3. "
            "Jarak ke ambang positif = lolos.",
            f"RM-a termurah (kasus terburuk): run #{int(cheapest['run_id'])}, lr={cheapest['lr']}, "
            f"epochs={cheapest['epochs']}, batch={cheapest['batch']}, warmup_ratio={cheapest['warmup_ratio']}, "
            f"weight_decay={cheapest['weight_decay']}, train_time_s={cheapest['train_time_s']}.",
        ]
        return TableData("4.18", "Evaluasi kriteria sukses", frame,
                         ["metrics/final_predictions.csv", "best.json", "runs_*.csv", "metrics/inference_benchmark.csv"], notes)

    def table_4_19(self) -> TableData:
        """Dimensi perbandingan langsung RM-b dan RM-c."""
        results = self.test_results()
        values = self.efficiency()
        outcome = self.criteria()
        rows = [
            {"dimensi": "val_f1_macro", "RM-b": float(self._champion_row("rmb")["val_f1_macro"]),
             "RM-c": float(self._champion_row("rmc")["val_f1_macro"])},
            {"dimensi": "test_f1_macro", "RM-b": results["rmb"]["f1_macro"], "RM-c": results["rmc"]["f1_macro"]},
            {"dimensi": "test_f1_judi", "RM-b": results["rmb"]["f1_class1"], "RM-c": results["rmc"]["f1_class1"]},
        ]
        for key in ("trainable_params", "train_time_s", "infer_peak_gpu_mem_mb", "infer_latency_ms", "index_size_mb"):
            rows.append({"dimensi": key, "RM-b": values["rmb"][key], "RM-c": values["rmc"][key]})
        rows.append({"dimensi": "kriteria_terpenuhi", "RM-b": outcome["rmb"]["kriteria_terpenuhi"],
                     "RM-c": outcome["rmc"]["kriteria_terpenuhi"]})
        frame = pd.DataFrame(rows)
        numeric = pd.to_numeric(frame["RM-c"], errors="coerce") - pd.to_numeric(frame["RM-b"], errors="coerce")
        frame["selisih_rmc_minus_rmb"] = numeric
        head_official = self._best()["rmc"].get("head_is_official_rmb", True)
        notes = ["RM-c memakai head RM-b resmi." if head_official
                 else "RM-c memakai head penantang yang BERBEDA dari head RM-b resmi; biaya latihnya milik head itu."]
        return TableData("4.19", "Perbandingan langsung RM-b dan RM-c", frame,
                         ["best.json", "runs_rmb.csv", "runs_rmc.csv", "metrics/final_predictions.csv",
                          "metrics/inference_benchmark.csv", "metrics/index_stats.json"], notes)

    def table_l_1(self) -> TableData:
        """Kontribusi RAC per head (lampiran)."""
        summary = self._rac_summary().sort_values("rmb_run_id").reset_index(drop=True)
        columns = ["rmb_run_id", "head_arch", "hidden_dim", "lr", "epochs", "trainable_params", "f1_no_rac",
                   "best_alpha", "best_k", "f1_best", "gain_best_pp", "f1_shared", "gain_shared_pp",
                   "f1_judi_no_rac", "gain_judi_best_pp", "shared_alpha", "shared_k"]
        return TableData("L.1", "Kontribusi RAC per head", summary[columns], ["runs_rmc.csv", "runs_rmb.csv"],
                         ["hidden_dim tidak bermakna untuk head linear. f1_no_rac = baris alpha = 0 (head dimuat dari checkpoint)."])

    # ------------------------------------------------------------------
    # Pembantu
    # ------------------------------------------------------------------

    def test_results(self) -> dict[str, dict[str, float]]:
        """Metrik test lengkap dan confusion matrix per skenario dari prediksi per baris."""
        predictions = self._metrics_csv("final_predictions.csv")
        results = {}
        for scenario in SCENARIO_LABELS:
            truth, pred = predictions["label"].to_numpy(), predictions[f"pred_{scenario}"].to_numpy()
            tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
            results[scenario] = {**self.evaluator.metrics(truth, pred),
                                 "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
        return results

    def _json(self, path: Path, name: str) -> dict:
        content = read_json(path, default=None)
        if not content:
            raise FileNotFoundError(f"{name} belum ada ({path})")
        return content

    def _runs(self, scenario: str) -> pd.DataFrame:
        frame = read_csv(self.out_dir / f"runs_{scenario}.csv")
        if frame.empty:
            raise FileNotFoundError(f"runs_{scenario}.csv belum ada atau kosong di {self.out_dir}")
        return frame

    def _best(self) -> dict[str, dict[str, object]]:
        return self._json(self.out_dir / "best.json", "best.json")

    def _metrics_csv(self, filename: str) -> pd.DataFrame:
        frame = read_csv(self.out_dir / "metrics" / filename)
        if frame.empty:
            raise FileNotFoundError(f"metrics/{filename} belum ada; jalankan 05_final_benchmark.ipynb lebih dulu")
        return frame

    def _champion_row(self, scenario: str) -> pd.Series:
        runs = self._runs(scenario)
        return runs[runs["run_id"] == int(self._best()[scenario]["run_id"])].iloc[0]

    def _rac_summary(self) -> pd.DataFrame:
        return rac_per_head(self._runs("rmc"), self._runs("rmb"))


__all__ = ["TableData", "ThesisDataExporter"]
