"""Test pencatatan riwayat run, kurva per-epoch, dan pelacakan juara."""

from __future__ import annotations

import pandas as pd
import pytest

from src.services.run_log import BestTracker, HistoryWriter, RunLogger
from src.utils.io import CorruptArtifactError


class TestRunLogger:
    def test_riwayat_kosong_dimulai_dari_satu(self, tmp_path) -> None:
        assert RunLogger(tmp_path / "runs_rma.csv").next_id() == 1

    def test_riwayat_menumpuk_lintas_instance(self, tmp_path) -> None:
        """Riwayat run harus bertahan antar sesi, bukan ditimpa."""
        path = tmp_path / "runs_rma.csv"
        RunLogger(path).log({"val_f1_macro": 0.90, "catatan": "run pertama"})
        RunLogger(path).log({"val_f1_macro": 0.92, "catatan": "run kedua"})

        rows = pd.read_csv(path)
        assert len(rows) == 2
        assert rows["run_id"].tolist() == [1, 2]
        assert rows["catatan"].tolist() == ["run pertama", "run kedua"]

    def test_run_id_dan_timestamp_ditambahkan(self, tmp_path) -> None:
        row = RunLogger(tmp_path / "runs_rma.csv").log({"val_f1_macro": 0.9})
        assert row["run_id"] == 1
        assert row["timestamp"]

    def test_delta_dihitung_terhadap_juara_sebelumnya(self, tmp_path) -> None:
        logger = RunLogger(tmp_path / "runs_rma.csv")
        logger.log({"val_f1_macro": 0.9000})
        row = logger.log({"val_f1_macro": 0.9150})
        assert row["delta_vs_best_f1_macro_pp"] == pytest.approx(1.5, abs=1e-6)

    def test_selisih_di_bawah_ambang_dianggap_seri(self, tmp_path) -> None:
        """Dengan satu seed, selisih sangat kecil bukan bukti keunggulan nyata."""
        logger = RunLogger(tmp_path / "runs_rma.csv", tie_threshold_pp=0.15)
        logger.log({"val_f1_macro": 0.9000})
        assert logger.log({"val_f1_macro": 0.9010})["is_tie_with_best"] is True

    def test_selisih_di_atas_ambang_bukan_seri(self, tmp_path) -> None:
        logger = RunLogger(tmp_path / "runs_rma.csv", tie_threshold_pp=0.15)
        logger.log({"val_f1_macro": 0.9000})
        assert logger.log({"val_f1_macro": 0.9100})["is_tie_with_best"] is False

    def test_run_pertama_tanpa_kolom_delta(self, tmp_path) -> None:
        row = RunLogger(tmp_path / "runs_rma.csv").log({"val_f1_macro": 0.9})
        assert "delta_vs_best_f1_macro_pp" not in row

    def test_sinyal_overfit_saat_epoch_terbaik_bukan_terakhir(self, tmp_path) -> None:
        logger = RunLogger(tmp_path / "runs_rma.csv")
        assert logger.log({"val_f1_macro": 0.9, "best_epoch": 3, "epochs": 5})[
            "overfit_signal"
        ] is True
        assert logger.log({"val_f1_macro": 0.9, "best_epoch": 5, "epochs": 5})[
            "overfit_signal"
        ] is False

    def test_best_f1_macro_mengambil_nilai_tertinggi(self, tmp_path) -> None:
        logger = RunLogger(tmp_path / "runs_rma.csv")
        for value in (0.90, 0.95, 0.93):
            logger.log({"val_f1_macro": value})
        assert logger.best_f1_macro() == pytest.approx(0.95)

    def test_csv_rusak_dilempar_bukan_menghapus_riwayat(self, tmp_path) -> None:
        """Perilaku lama menelan galat lalu menulis ulang berkas dari riwayat kosong.

        Satu CSV yang sedang terbuka di Excel cukup untuk menghapus 26 run.
        """
        path = tmp_path / "runs_rma.csv"
        path.write_text('a,b\n1,2\n3,4,5,6\n"tak ditutup', encoding="utf-8")
        with pytest.raises(CorruptArtifactError):
            RunLogger(path)

    def test_konfigurasi_gagal_dicatat_terpisah(self, tmp_path) -> None:
        logger = RunLogger(tmp_path / "runs_rmb.csv")
        path = logger.log_error(
            "rmb", "batch_1", 3, {"head_arch": "cnn"}, "sengaja salah",
            ValueError("arsitektur tak dikenal"),
        )
        errors = pd.read_csv(path)
        assert path.name == "runs_rmb_errors.csv"
        assert errors.loc[0, "error_type"] == "ValueError"
        assert errors.loc[0, "seq"] == 3
        assert errors.loc[0, "head_arch"] == "cnn"

    def test_galat_juga_menumpuk(self, tmp_path) -> None:
        logger = RunLogger(tmp_path / "runs_rmb.csv")
        for seq in (1, 2):
            logger.log_error("rmb", "b", seq, {}, "", RuntimeError("gagal"))
        assert len(pd.read_csv(tmp_path / "runs_rmb_errors.csv")) == 2


class TestHistoryWriter:
    def test_menulis_kurva_dengan_kolom_run_id(self, tmp_path) -> None:
        writer = HistoryWriter(tmp_path)
        path = writer.append("rma", 1, [{"epoch": 1, "val_f1_macro": 0.9}])
        frame = pd.read_csv(path)
        assert frame.columns[0] == "run_id"
        assert frame["run_id"].tolist() == [1]

    def test_run_berbeda_menumpuk_di_satu_berkas(self, tmp_path) -> None:
        writer = HistoryWriter(tmp_path)
        writer.append("rma", 1, [{"epoch": 1, "val_f1_macro": 0.90}])
        writer.append("rma", 2, [{"epoch": 1, "val_f1_macro": 0.92}])
        assert sorted(pd.read_csv(writer.path("rma"))["run_id"]) == [1, 2]

    def test_run_id_yang_sama_ditimpa_bukan_diduplikasi(self, tmp_path) -> None:
        writer = HistoryWriter(tmp_path)
        writer.append("rma", 1, [{"epoch": 1, "val_f1_macro": 0.90}])
        writer.append("rma", 1, [{"epoch": 1, "val_f1_macro": 0.95}])
        frame = pd.read_csv(writer.path("rma"))
        assert len(frame) == 1
        assert frame.loc[0, "val_f1_macro"] == pytest.approx(0.95)

    def test_history_kosong_tidak_membuat_baris(self, tmp_path) -> None:
        writer = HistoryWriter(tmp_path)
        path = writer.append("rma", 1, [])
        assert not path.exists()


class TestBestTracker:
    def test_skenario_belum_ada_mengembalikan_dict_kosong(self, tmp_path) -> None:
        assert BestTracker(tmp_path).get("rma") == {}

    def test_run_pertama_selalu_menjadi_juara(self, tmp_path) -> None:
        tracker = BestTracker(tmp_path)
        assert tracker.update("rma", {"run_id": 1, "val_f1_macro": 0.90}) is True

    def test_run_lebih_baik_menggantikan_juara(self, tmp_path) -> None:
        tracker = BestTracker(tmp_path)
        tracker.update("rma", {"run_id": 1, "val_f1_macro": 0.90})
        assert tracker.update("rma", {"run_id": 2, "val_f1_macro": 0.93}) is True
        assert tracker.get("rma")["run_id"] == 2

    def test_run_lebih_buruk_tidak_menggantikan(self, tmp_path) -> None:
        tracker = BestTracker(tmp_path)
        tracker.update("rma", {"run_id": 1, "val_f1_macro": 0.93})
        assert tracker.update("rma", {"run_id": 2, "val_f1_macro": 0.90}) is False
        assert tracker.get("rma")["run_id"] == 1

    def test_nilai_sama_tidak_menggantikan(self, tmp_path) -> None:
        """Seri diselesaikan oleh aturan seleksi, bukan oleh urutan kedatangan."""
        tracker = BestTracker(tmp_path)
        tracker.update("rma", {"run_id": 1, "val_f1_macro": 0.93})
        assert tracker.update("rma", {"run_id": 2, "val_f1_macro": 0.93}) is False

    def test_skenario_terpisah_tidak_saling_menimpa(self, tmp_path) -> None:
        tracker = BestTracker(tmp_path)
        tracker.update("rma", {"run_id": 1, "val_f1_macro": 0.98})
        tracker.update("rmb", {"run_id": 1, "val_f1_macro": 0.96})
        assert set(BestTracker(tmp_path).data) == {"rma", "rmb"}

    def test_juara_bertahan_lintas_instance(self, tmp_path) -> None:
        BestTracker(tmp_path).update("rma", {"run_id": 7, "val_f1_macro": 0.97})
        assert BestTracker(tmp_path).get("rma")["run_id"] == 7

    def test_replace_mengganti_juara_walau_f1_lebih_rendah(self, tmp_path) -> None:
        """Putusan dengan aturan lain (ambang seri, bootstrap) tidak dinilai F1 semata."""
        tracker = BestTracker(tmp_path)
        tracker.update("rmc", {"run_id": 1, "val_f1_macro": 0.97})
        tracker.replace("rmc", {"run_id": None, "val_f1_macro": 0.96, "source": "eksplorasi"})
        assert tracker.get("rmc")["source"] == "eksplorasi"

    def test_replace_tertulis_ke_berkas(self, tmp_path) -> None:
        BestTracker(tmp_path).replace("rmc", {"run_id": None, "val_f1_macro": 0.96})
        assert BestTracker(tmp_path).get("rmc")["val_f1_macro"] == 0.96

    def test_replace_tanpa_f1_ditolak(self, tmp_path) -> None:
        with pytest.raises(KeyError, match="val_f1_macro"):
            BestTracker(tmp_path).replace("rmc", {"run_id": 1})
