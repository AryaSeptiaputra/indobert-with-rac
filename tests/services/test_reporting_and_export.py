"""Test pelaporan visual, penggabungan riwayat, ekspor Excel, dan benchmark FAISS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.services.aggregation import RunMerger
from src.services.faiss_benchmark import FaissBenchmark
from src.services.reporting import FigureReporter
from src.services.workbook import WorkbookBuilder
from src.utils.io import write_csv, write_json


@pytest.fixture
def campaign_dir(tmp_path, rng):
    """Folder keluaran kampanye tiruan dengan grid RM-a dua dimensi."""
    rows = []
    run_id = 0
    for lr in (1e-5, 2e-5, 3e-5):
        for epochs in (3, 5):
            for batch in (16, 32):
                run_id += 1
                rows.append(
                    {
                        "run_id": run_id,
                        "scenario": "rma",
                        "lr": lr,
                        "epochs": epochs,
                        "batch": batch,
                        "val_f1_macro": float(rng.uniform(0.90, 0.98)),
                        "train_time_s": float(rng.uniform(80, 220)),
                        "trainable_params": 109_485_314,
                        "peak_mem_mb": float(rng.uniform(2000, 3000)),
                        "catatan": "sel grid",
                    }
                )
    write_csv(tmp_path / "runs_rma.csv", pd.DataFrame(rows))
    write_json(
        tmp_path / "best.json",
        {"rma": {"run_id": 1, "val_f1_macro": 0.97, "config": {"lr": 2e-5}}},
    )
    write_json(tmp_path / "hardware.json", {"gpu": "RTX 3050 Laptop", "torch": "2.12.1"})
    return tmp_path


class TestFigureReporter:
    def test_kurva_training_dibuat(self, campaign_dir) -> None:
        path = FigureReporter(campaign_dir).training_curve(
            "rma", 1,
            [
                {"epoch": 1, "train_loss": 0.5, "val_f1_macro": 0.90, "val_f1_judi": 0.80},
                {"epoch": 2, "train_loss": 0.3, "val_f1_macro": 0.94, "val_f1_judi": 0.88},
            ],
        )
        assert path is not None and path.exists()

    def test_history_kosong_tidak_membuat_gambar(self, campaign_dir) -> None:
        assert FigureReporter(campaign_dir).training_curve("rma", 1, []) is None

    def test_history_tanpa_kolom_inti_dilewati(self, campaign_dir) -> None:
        reporter = FigureReporter(campaign_dir)
        assert reporter.training_curve("rma", 1, [{"epoch": 1}]) is None

    def test_heatmap_grid_satu_gambar_per_facet(self, campaign_dir) -> None:
        """Grid dibaca sebagai permukaan: satu pivot lr x epochs per nilai batch."""
        written = FigureReporter(campaign_dir).grid_pivot_and_heatmap("rma")
        heatmaps = [p for p in written if p.suffix == ".png"]
        assert len(heatmaps) == 2
        assert all(p.exists() for p in written)

    def test_heatmap_dilewati_saat_variasi_kurang(self, tmp_path) -> None:
        write_csv(
            tmp_path / "runs_rma.csv",
            pd.DataFrame([{"run_id": 1, "lr": 2e-5, "epochs": 5, "batch": 16,
                           "val_f1_macro": 0.9}]),
        )
        assert FigureReporter(tmp_path).grid_pivot_and_heatmap("rma") == []

    def test_scatter_tradeoff_dibuat(self, campaign_dir) -> None:
        path = FigureReporter(campaign_dir).tradeoff_scatter("rma")
        assert path is not None and path.exists()

    def test_scatter_butuh_minimal_dua_baris(self, tmp_path) -> None:
        write_csv(
            tmp_path / "runs_rma.csv",
            pd.DataFrame([{"run_id": 1, "val_f1_macro": 0.9, "train_time_s": 80.0}]),
        )
        assert FigureReporter(tmp_path).tradeoff_scatter("rma") is None

    def test_bar_chart_top_config(self, campaign_dir) -> None:
        path = FigureReporter(campaign_dir).top_configs_bar_chart("rma", top_n=5)
        assert path is not None and path.exists()

    def test_riwayat_kosong_tidak_membuat_bar_chart(self, tmp_path) -> None:
        assert FigureReporter(tmp_path).top_configs_bar_chart("rma") is None

    def test_confusion_matrix_dan_pr_curve(self, campaign_dir, rng) -> None:
        reporter = FigureReporter(campaign_dir)
        y_true = rng.integers(0, 2, 100)
        y_score = np.clip(y_true * 0.6 + rng.normal(0.2, 0.2, 100), 0, 1)
        assert reporter.confusion_matrix(
            y_true, (y_score > 0.5).astype(int), "cm.png"
        ).exists()
        assert reporter.pr_curve(y_true, y_score, "pr.png").exists()

    def test_ringkasan_memuat_jumlah_run_dan_juara(self, campaign_dir) -> None:
        import json

        path = FigureReporter(campaign_dir).write_summary()
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["scenarios"]["rma"]["n_runs"] == 12
        assert summary["scenarios"]["rma"]["best_run_id"] == 1

    def test_refresh_membuat_seluruh_gambar_agregat(self, campaign_dir) -> None:
        written = FigureReporter(campaign_dir).refresh_scenario("rma")
        assert len(written) >= 4
        assert all(path.exists() for path in written)


class TestRunMerger:
    def test_kolom_source_ditambahkan_di_depan(self, tmp_path) -> None:
        for name in ("A", "B"):
            directory = tmp_path / name
            directory.mkdir()
            write_csv(
                directory / "runs_rma.csv",
                pd.DataFrame([{"run_id": 1, "val_f1_macro": 0.9}]),
            )
        merged = RunMerger({"A": tmp_path / "A", "B": tmp_path / "B"}).merge_scenario("rma")
        assert merged.columns[0] == "source"
        assert sorted(merged["source"]) == ["A", "B"]

    def test_run_id_berulang_lintas_source(self, tmp_path) -> None:
        """Kunci baris adalah pasangan (source, run_id); run_id sendiri tidak unik."""
        for name in ("A", "B"):
            directory = tmp_path / name
            directory.mkdir()
            write_csv(
                directory / "runs_rma.csv",
                pd.DataFrame([{"run_id": 1, "val_f1_macro": 0.9}]),
            )
        merged = RunMerger({"A": tmp_path / "A", "B": tmp_path / "B"}).merge_scenario("rma")
        assert merged["run_id"].tolist() == [1, 1]
        assert len(merged.drop_duplicates(subset=["source", "run_id"])) == 2

    def test_sumber_kosong_dilewati(self, tmp_path) -> None:
        (tmp_path / "kosong").mkdir()
        assert RunMerger({"K": tmp_path / "kosong"}).merge_scenario("rma").empty

    def test_merge_all_menulis_readme_peringatan(self, tmp_path) -> None:
        source = tmp_path / "A"
        source.mkdir()
        write_csv(source / "runs_rma.csv", pd.DataFrame([{"run_id": 1, "val_f1_macro": 0.9}]))
        written = RunMerger({"A": source}).merge_all(tmp_path / "combined")
        readme = written["readme"].read_text(encoding="utf-8")
        assert "tidak boleh" in readme
        assert "satu hardware" in readme

    def test_berkas_sumber_tidak_diubah(self, tmp_path) -> None:
        source = tmp_path / "A"
        source.mkdir()
        path = write_csv(source / "runs_rma.csv", pd.DataFrame([{"run_id": 1, "val_f1_macro": 0.9}]))
        before = path.read_bytes()
        RunMerger({"A": source}).merge_all(tmp_path / "combined")
        assert path.read_bytes() == before


class TestWorkbookBuilder:
    def test_membuat_workbook_dengan_sheet_ringkasan(self, campaign_dir) -> None:
        import openpyxl

        path = WorkbookBuilder(campaign_dir).build()
        assert path.exists()
        workbook = openpyxl.load_workbook(path, read_only=True)
        try:
            assert workbook.sheetnames[0] == "Ringkasan"
            assert "Run RMA" in workbook.sheetnames
        finally:
            workbook.close()

    def test_ringkasan_memuat_konteks_hardware(self, campaign_dir) -> None:
        """Angka efisiensi tak bisa ditafsirkan tanpa tahu di mana diukur."""
        summary = WorkbookBuilder(campaign_dir).summary_sheet()
        assert "hardware.gpu" in summary["keterangan"].tolist()

    def test_folder_kosong_tetap_menghasilkan_ringkasan(self, tmp_path) -> None:
        summary = WorkbookBuilder(tmp_path).summary_sheet()
        assert summary.loc[0, "nilai"] == "belum ada artefak kampanye"

    def test_nama_sheet_dipotong_ke_batas_excel(self) -> None:
        assert len(WorkbookBuilder._sheet_name("x" * 50)) == 31


class TestFaissBenchmark:
    @pytest.fixture
    def embeddings(self, rng) -> tuple[np.ndarray, np.ndarray]:
        return (
            rng.normal(size=(300, 32)).astype(np.float32),
            rng.integers(0, 2, 300),
        )

    def test_jumlah_label_harus_cocok(self, rng) -> None:
        with pytest.raises(ValueError, match="jumlah embedding"):
            FaissBenchmark(rng.normal(size=(10, 4)), np.zeros(5))

    def test_metrik_bangun_indeks(self, embeddings) -> None:
        train, labels = embeddings
        build = FaissBenchmark(train, labels, repeats=2).measure_build()
        assert build["n_vectors"] == 300
        assert build["dim"] == 32
        assert build["index_ntotal"] == 300
        assert build["build_time_ms_mean"] >= 0.0

    def test_ukuran_indeks_sesuai_rumus_float32(self, embeddings) -> None:
        """Indeks flat menyimpan seluruh vektor apa adanya.

        Toleransi mengikuti pembulatan tiga desimal pada nilai yang dilaporkan.
        """
        train, labels = embeddings
        build = FaissBenchmark(train, labels, repeats=1).measure_build()
        assert build["index_size_mb"] == pytest.approx(300 * 32 * 4 / 1024**2, abs=5e-4)

    def test_waktu_telusur_naik_mengikuti_k(self, embeddings) -> None:
        train, labels = embeddings
        rows = FaissBenchmark(train, labels, repeats=3).measure_search(
            train[:50], k_values=(1, 5, 20)
        )
        assert [row["k"] for row in rows] == [1, 5, 20]
        assert all(row["n_queries"] == 50 for row in rows)

    def test_k_melebihi_indeks_ditolak(self, embeddings) -> None:
        train, labels = embeddings
        with pytest.raises(ValueError, match="melebihi"):
            FaissBenchmark(train, labels, repeats=1).measure_search(
                train[:10], k_values=(500,)
            )

    def test_run_menulis_kedua_artefak(self, embeddings, tmp_path) -> None:
        train, labels = embeddings
        FaissBenchmark(train, labels, repeats=2).run(
            train[:20], k_values=(1, 5), out_dir=tmp_path
        )
        assert (tmp_path / "faiss_build.json").exists()
        assert (tmp_path / "faiss_search.csv").exists()
