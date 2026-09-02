"""Test integrasi CampaignRunner untuk RM-b dan RM-c.

RM-a sengaja tidak diuji di sini: skenario itu memuat bobot IndoBERT 110 juta
parameter dari Hub, terlalu berat dan terlalu bergantung jaringan untuk sebuah
test. Fitur beku disuntikkan langsung agar keseluruhan alur pencatatan tetap
teruji tanpa memanggil encoder.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.services.campaign import CampaignRunner
from src.services.data import ExperimentData
from src.services.features import FeatureSet


@pytest.fixture
def runner(tmp_path, split_frames, feature_set, class_weights, cpu_device, monkeypatch):
    """CampaignRunner dengan data dan fitur sintetis, tanpa menyentuh Hub."""
    campaign = CampaignRunner(out_dir=tmp_path, device="cpu")
    campaign._data = ExperimentData(
        frames=split_frames,
        tokenizer=None,
        class_weights=class_weights,
        device=cpu_device,
        model_name="sintetis",
        max_length=128,
    )
    campaign._features = feature_set
    return campaign


class TestRunTunggal:
    def test_skenario_tak_dikenal_ditolak(self, runner) -> None:
        with pytest.raises(ValueError, match="skenario tak dikenal"):
            runner.run("rmd")

    def test_rmb_mencatat_satu_baris(self, runner, tmp_path) -> None:
        row = runner.run("rmb", {"epochs": 2}, note="uji integrasi")
        assert row["run_id"] == 1
        assert row["catatan"] == "uji integrasi"
        assert len(pd.read_csv(tmp_path / "runs_rmb.csv")) == 1

    def test_konfigurasi_invalid_ditolak_sebelum_training(self, runner) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            runner.run("rmb", {"head_arch": "cnn"})

    def test_waktu_latih_rmb_mencakup_ekstraksi_fitur(self, runner) -> None:
        """Biaya ekstraksi adalah bagian jujur dari waktu latih RM-b."""
        row = runner.run("rmb", {"epochs": 1})
        assert row["extract_time_s"] == pytest.approx(1.5)
        assert row["train_time_s"] == pytest.approx(
            row["head_train_time_s"] + row["extract_time_s"], abs=0.011
        )

    def test_rmb_menyimpan_checkpoint_juara(self, runner, tmp_path) -> None:
        runner.run("rmb", {"epochs": 2})
        checkpoint = tmp_path / "checkpoints" / "rmb_best.pt"
        assert checkpoint.exists()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert set(payload) >= {"head_state", "config", "run_id", "hidden_size", "model_name"}

    def test_test_set_tidak_disentuh_secara_default(self, runner, tmp_path) -> None:
        """Membuka test saat tuning akan membocorkan pemilihan model."""
        runner.run("rmb", {"epochs": 1})
        columns = pd.read_csv(tmp_path / "runs_rmb.csv").columns
        assert not any(column.startswith("test_") for column in columns)

    def test_eval_test_menambahkan_kolom_test(self, runner, tmp_path) -> None:
        runner.run("rmb", {"epochs": 1}, eval_test=True)
        columns = pd.read_csv(tmp_path / "runs_rmb.csv").columns
        assert "test_f1_macro" in columns


class TestRunBatch:
    def test_seluruh_konfigurasi_dijalankan(self, runner, tmp_path) -> None:
        frame = runner.run_batch(
            "rmb",
            [
                {"config": {"epochs": 1, "lr": 1e-3}, "note": "lr tinggi"},
                {"config": {"epochs": 1, "lr": 1e-4}, "note": "lr rendah"},
            ],
        )
        assert len(frame) == 2
        assert frame["run_id"].tolist() == [1, 2]

    def test_kegagalan_diisolasi_dan_batch_lanjut(self, runner, tmp_path) -> None:
        """Kampanye lima jam tidak boleh batal karena satu nilai keliru."""
        frame = runner.run_batch(
            "rmb",
            [
                {"config": {"epochs": 1}, "note": "sah"},
                {"config": {"head_arch": "cnn"}, "note": "sengaja salah"},
                {"config": {"epochs": 1, "lr": 5e-4}, "note": "sah juga"},
            ],
        )
        assert len(frame) == 2
        errors = pd.read_csv(tmp_path / "runs_rmb_errors.csv")
        assert len(errors) == 1
        assert errors.loc[0, "seq"] == 2

    def test_batch_id_dicatat_di_tiap_baris(self, runner) -> None:
        frame = runner.run_batch(
            "rmb", [{"config": {"epochs": 1}}], batch_id="grid_tahap1"
        )
        assert frame.loc[0, "batch_id"] == "grid_tahap1"

    def test_batch_menghasilkan_ringkasan(self, runner, tmp_path) -> None:
        runner.run_batch("rmb", [{"config": {"epochs": 1}}])
        assert (tmp_path / "tuning_summary.json").exists()


class TestRMC:
    def test_rmc_butuh_head_rmb(self, runner) -> None:
        with pytest.raises(FileNotFoundError, match="rmb_best.pt"):
            runner.run("rmc", {"alpha": 0.2})

    def test_rmc_tanpa_parameter_dan_tanpa_waktu_latih(self, runner) -> None:
        runner.run("rmb", {"epochs": 2})
        row = runner.run("rmc", {"alpha": 0.2, "k": 5}, note="sweep alpha")
        assert row["trainable_params"] == 0
        assert row["train_time_s"] == 0.0

    def test_rmc_memakai_indeks_seukuran_split_train(self, runner, feature_set) -> None:
        runner.run("rmb", {"epochs": 2})
        row = runner.run("rmc", {"alpha": 0.2})
        assert row["index_vectors"] == len(feature_set.labels["train"])

    def test_rmc_mencatat_arsitektur_head_yang_diwarisi(self, runner) -> None:
        runner.run("rmb", {"head_arch": "mlp", "hidden_dim": 64, "epochs": 2})
        row = runner.run("rmc", {"alpha": 0.2})
        assert row["head_arch_rmb"] == "mlp"

    def test_penomoran_run_terpisah_per_skenario(self, runner) -> None:
        runner.run("rmb", {"epochs": 1})
        runner.run("rmb", {"epochs": 1})
        assert runner.run("rmc", {"alpha": 0.1})["run_id"] == 1


class TestFinal:
    def test_final_menolak_skenario_yang_belum_dijalankan(self, runner) -> None:
        runner.run("rmb", {"epochs": 1})
        with pytest.raises(RuntimeError, match="belum punya run"):
            runner.run_final()
