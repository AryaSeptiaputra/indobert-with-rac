"""Test pengekspor data mentah Tabel 4.1-4.19 dan L.1."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.services.candidates import build_candidates, write_candidates
from src.services.thesis_data import ThesisDataExporter
from tests.services.test_thesis_figures import RMC_CHAMPION_HEAD, tulis_log_kampanye


def lengkapi_log(out_dir, tmp_path) -> dict:
    """Tambah berkas yang dibutuhkan tabel di atas log sintetis gambar."""
    tulis_log_kampanye(out_dir)

    rma = pd.read_csv(out_dir / "runs_rma.csv").assign(
        warmup_ratio=0.1, weight_decay=0.01, micro_batch=32, seed=42, best_epoch=4,
        val_f1_judi=0.95, val_acc=0.98, val_precision_macro=0.97, val_recall_macro=0.96,
    )
    rma.loc[rma["run_id"] == 3, "train_time_s"] = 500.0
    rma.to_csv(out_dir / "runs_rma.csv", index=False)

    rmb = pd.read_csv(out_dir / "runs_rmb.csv").assign(
        seed=42, val_acc=0.97, val_precision_macro=0.96, val_recall_macro=0.95,
    )
    rmb.to_csv(out_dir / "runs_rmb.csv", index=False)

    rmc = pd.read_csv(out_dir / "runs_rmc.csv")
    params = rmb.set_index("run_id")
    rmc = rmc.assign(
        batch_id="rmc_grid_seluruh_head", index_vectors=6588, index_type="IndexFlatIP",
        val_acc=0.98, val_precision_macro=0.97, val_recall_macro=0.96,
        head_trainable_params=rmc["rmb_run_id"].map(params["trainable_params"]),
        head_train_time_s=rmc["rmb_run_id"].map(params["train_time_s"]),
    )
    rmc.to_csv(out_dir / "runs_rmc.csv", index=False)

    best = json.loads((out_dir / "best.json").read_text(encoding="utf-8"))
    final_rmc = rmc[(rmc["rmb_run_id"] == RMC_CHAMPION_HEAD) & (rmc["alpha"] == 0.2) & (rmc["k"] == 5)].iloc[0]
    best["rmc"].update(run_id=int(final_rmc["run_id"]), decided=True, head_is_official_rmb=False)
    best["rmb"]["config"] = {}
    best["rma"]["run_id"] = 1
    (out_dir / "best.json").write_text(json.dumps(best), encoding="utf-8")

    summary = lambda row: {"run_id": int(row["run_id"]), "rmb_run_id": int(row["rmb_run_id"])}  # noqa: E731
    (out_dir / "rmc_champion_decision.json").write_text(json.dumps({
        "rule": "aturan", "official_rmb_run_id": 6, "winner": "penantang", "head_is_official_rmb": False,
        "default": summary(rmc[rmc["rmb_run_id"] == 6].iloc[0]), "challenger": summary(final_rmc),
        "bootstrap": {"observed_delta_pp": 0.3, "mean_delta_pp": 0.29, "ci_low_pp": 0.05, "ci_high_pp": 0.6,
                      "n_boot": 10000, "seed": 42, "n_samples": 1402},
        "cost_note": "biaya pelatihan RM-c = biaya head penantang #4, BUKAN warisan head RM-b resmi",
    }), encoding="utf-8")

    body = build_candidates(
        {s: pd.read_csv(out_dir / f"runs_{s}.csv") for s in ("rma", "rmb", "rmc")},
        {"rma": 1, "rmb": best["rmb"]["run_id"], "rmc": int(final_rmc["run_id"])},
    )
    stamped = write_candidates(out_dir / "candidates.json", body)
    rows = []
    for scenario, entry in stamped["scenarios"].items():
        for rank, key in ((1, "first"), (2, "second")):
            if entry.get(key):
                rows.append({"scenario": scenario, "rank": rank, "run_id": entry[key]["run_id"],
                             "test_f1_macro": 0.96 - rank * 0.001, "candidates_sha256": stamped["content_sha256"]})
    pd.DataFrame(rows).to_csv(out_dir / "metrics" / "candidates_test.csv", index=False)

    pd.DataFrame({"scenario": ["RM-a", "RM-b", "RM-c"], "infer_latency_ms": [5.7, 5.6, 5.8],
                  "infer_peak_gpu_mem_mb": [900.0, 850.0, 856.0], "latency_warmup_runs": 10,
                  "latency_runs": 100}).to_csv(out_dir / "metrics" / "inference_benchmark.csv", index=False)
    (out_dir / "metrics" / "index_stats.json").write_text(json.dumps({
        "index_type": "IndexFlatIP", "vectors": 6588, "dimension": 768, "size_mb": 19.3,
        "build_time_s": 0.012, "source_split": "train"}), encoding="utf-8")
    (out_dir / "hardware.json").write_text(json.dumps({
        "gpu": "RTX 3090", "vram_total_mb": 24576, "driver": "550", "cuda_version": "12.4", "cpu": "EPYC",
        "cpu_logical_cores": 32, "ram_total_gb": 125.7, "os": "Linux", "python": "3.13.1", "torch": "2.12.1",
        "transformers": "5.12.1", "faiss": "1.14.3", "seed": 42,
        "tuning_sessions": [{"recorded_at": "2026-09-24 02:00:00", "boot_time": "2026-09-24 01:00:00"}],
        "final_session": {"recorded_at": "2026-09-24 09:00:00", "boot_time": "2026-09-24 01:00:00",
                          "same_boot_as_tuning": True},
    }), encoding="utf-8")

    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "counts": {"raw": 14237, "after_missing": 14227, "after_dedup": 9412, "leakage_removed": 17, "final_total": 9395},
        "label_conflict": {"groups": 3, "rows": 32, "policy": "assign_1"},
        "splits": {"train": {"n": 6588, "L0": 5391, "L1": 1197}, "val": {"n": 1402, "L0": 1145, "L1": 257},
                   "test": {"n": 1405, "L0": 1149, "L1": 256}},
        "class_weights": {"0": 0.6110183639398998, "1": 2.7518796992481205},
    }), encoding="utf-8")
    stats = tmp_path / "preprocessing_stats.json"
    stats.write_text(json.dumps({
        "leakage_removed_by_class": {"judi": 5, "non_judi": 12},
        "nfkc_unk": {"unit": "persen baris", "n_rows": {"judi": 1710, "non_judi": 7685},
                     "rates_pct": {"sebelum_nfkc": {"judi": 12.5, "non_judi": 3.1},
                                   "sesudah_nfkc": {"judi": 2.4, "non_judi": 1.9}}},
    }), encoding="utf-8")
    return {"metadata_path": metadata, "preprocessing_stats_path": stats}


@pytest.fixture
def exporter(tmp_path):
    out_dir = tmp_path / "tuning"
    out_dir.mkdir()
    paths = lengkapi_log(out_dir, tmp_path)
    return ThesisDataExporter(out_dir, **paths)


class TestEkspor:
    def test_seluruh_tabel_tersedia_dan_ditulis_ke_xlsx(self, exporter) -> None:
        results = exporter.collect_all(skip_missing=False)
        assert list(results) == [f"4.{i}" for i in range(1, 20)] + ["L.1"]
        path = exporter.export(results)
        sheets = pd.ExcelFile(path).sheet_names
        assert sheets[0] == "keterangan" and "4.17" in sheets and "L.1" in sheets
        keterangan = pd.read_excel(path, sheet_name="keterangan")
        assert (keterangan["status"] == "tersedia").all()

    def test_tanpa_hasil_final_tabel_test_dilewati(self, exporter) -> None:
        import shutil

        shutil.rmtree(exporter.out_dir / "metrics")
        results = exporter.collect_all()
        assert all(str(results[n]).startswith("dilewati") for n in ("4.14", "4.15", "4.17", "4.18", "4.19"))
        assert all(not isinstance(results[n], str) for n in ("4.1", "4.8", "4.11", "4.12", "L.1"))


class TestIsiTabel:
    def test_4_2_memuat_baris_terbuang_per_kelas(self, exporter) -> None:
        frame = exporter.table_4_2().frame.set_index("tahap")
        assert frame.loc["guard anti-leakage", "perubahan_baris"] == -17
        assert frame.loc["guard: terbuang kelas judi", "jumlah_baris"] == 5

    def test_4_7_vektor_indeks_sama_dengan_baris_train(self, exporter) -> None:
        frame = exporter.table_4_7().frame
        row = frame[frame["parameter"] == "sama_dengan_train"].iloc[0]
        assert row["nilai"] == "True"

    def test_4_11_mencatat_peran_dan_bootstrap(self, exporter) -> None:
        data = exporter.table_4_11()
        assert data.frame["peran"].str.contains("final").any()
        assert data.frame["peran"].str.contains("default").any()
        assert any("n_boot=10000" in note for note in data.notes)
        assert any("BUKAN warisan" in note for note in data.notes)

    def test_4_12_kelompok_menurut_arsitektur(self, exporter) -> None:
        frame = exporter.table_4_12().frame
        assert frame["kelompok_head"].tolist()[0] == "Linear"
        assert frame.set_index("kelompok_head").loc["Linear", "jumlah_head"] == 1

    def test_4_14_confusion_matrix_konsisten(self, exporter) -> None:
        frame = exporter.table_4_14().frame.set_index("metrik")
        assert frame.loc["tp", "RM-a"] + frame.loc["fn", "RM-a"] == 255

    def test_4_17_rmc_memakai_biaya_head_penantang(self, exporter, tmp_path) -> None:
        frame = exporter.table_4_17().frame.set_index("metrik")
        head = pd.read_csv(tmp_path / "tuning" / "runs_rmb.csv").set_index("run_id").loc[RMC_CHAMPION_HEAD]
        assert frame.loc["head_rmb_run_id", "RM-c"] == RMC_CHAMPION_HEAD
        assert frame.loc["trainable_params", "RM-c"] == head["trainable_params"]
        assert frame.loc["index_build_s", "RM-c"] == 0.012

    def test_4_18_rma_termurah_seluruh_grid(self, exporter, tmp_path) -> None:
        data = exporter.table_4_18()
        runs = pd.read_csv(tmp_path / "tuning" / "runs_rma.csv")
        cheapest = runs.sort_values(["train_time_s", "run_id"]).iloc[0]
        assert f"run #{int(cheapest['run_id'])}" in data.notes[1]
        assert set(data.frame["skenario"]) == {"RM-b", "RM-c"}

    def test_4_16_hash_tidak_cocok_ditolak(self, exporter, tmp_path) -> None:
        path = tmp_path / "tuning" / "metrics" / "candidates_test.csv"
        frame = pd.read_csv(path).assign(candidates_sha256="palsu")
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="hash"):
            exporter.table_4_16()
