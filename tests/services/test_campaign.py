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
from src.services.fusion_ablation import FusionFormulaConfig


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
        with pytest.raises(FileNotFoundError, match="juara RM-b belum ada"):
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

class TestMemoriPelatihanRMB:
    def test_memori_pelatihan_mencakup_ekstraksi(self, runner, feature_set) -> None:
        """Waktu latih RM-b mencakup ekstraksi, jadi memori puncaknya juga."""
        feature_set.extract_peak_mem_mb = 850.0
        row = runner.run("rmb", {"epochs": 1})
        assert row["extract_peak_mem_mb"] == 850.0
        assert row["train_peak_mem_mb"] == max(row["peak_mem_mb"], 850.0)


class TestResume:
    """Batch yang terputus harus bisa dijalankan ulang tanpa mengulang pekerjaan."""

    def test_tanda_pengenal_melengkapi_nilai_default(self, runner) -> None:
        parsial = runner.config_signature("rmb", {"epochs": 3})
        lengkap = runner.config_signature(
            "rmb",
            {"head_arch": "linear", "hidden_dim": 256, "epochs": 3, "lr": 2e-4,
             "dropout": 0.1, "weight_decay": 0.01, "batch": 32, "seed": 42},
        )
        assert parsial == lengkap

    def test_konfigurasi_berbeda_bertanda_berbeda(self, runner) -> None:
        assert runner.config_signature("rmb", {"epochs": 3}) != runner.config_signature(
            "rmb", {"epochs": 4}
        )

    def test_micro_batch_dinormalkan_ke_nilai_efektif(self, runner) -> None:
        """micro_batch 32 pada batch 16 menghasilkan run yang sama dengan 16."""
        assert runner.config_signature(
            "rma", {"batch": 16, "micro_batch": 32}
        ) == runner.config_signature("rma", {"batch": 16, "micro_batch": 16})

    def test_micro_batch_efektif_berbeda_bertanda_berbeda(self, runner) -> None:
        """DataLoader dengan batch 8 dan 16 mengonsumsi RNG berbeda, jadi hasilnya beda."""
        assert runner.config_signature(
            "rma", {"batch": 16, "micro_batch": 8}
        ) != runner.config_signature("rma", {"batch": 16, "micro_batch": 16})

    def test_riwayat_kosong_tidak_melewati_apa_pun(self, runner) -> None:
        assert runner.completed_signatures("rmb") == set()
        requests = [{"config": {"epochs": 1}}, {"config": {"epochs": 2}}]
        assert len(runner.pending_requests("rmb", requests)) == 2

    def test_run_yang_sudah_ada_terdeteksi(self, runner) -> None:
        runner.run("rmb", {"epochs": 2}, note="run pertama")
        assert runner.config_signature("rmb", {"epochs": 2}) in runner.completed_signatures(
            "rmb"
        )

    def test_pending_menyaring_yang_sudah_dijalankan(self, runner) -> None:
        runner.run("rmb", {"epochs": 1})
        pending = runner.pending_requests(
            "rmb", [{"config": {"epochs": 1}}, {"config": {"epochs": 2}}]
        )
        assert len(pending) == 1
        assert pending[0].config["epochs"] == 2

    def test_pending_membuang_kembar_di_dalam_permintaan(self, runner) -> None:
        pending = runner.pending_requests(
            "rmb",
            [{"config": {"epochs": 1}}, {"config": {"epochs": 1}}, {"config": {"epochs": 2}}],
        )
        assert len(pending) == 2

    def test_batch_melanjutkan_dari_yang_terakhir(self, runner, tmp_path) -> None:
        """Skenario nyata: batch terputus di tengah, lalu dijalankan ulang apa adanya."""
        grid = [{"config": {"epochs": e}, "note": f"sel epochs={e}"} for e in (1, 2, 3)]

        runner.run_batch("rmb", grid[:2], batch_id="percobaan_pertama")
        assert len(pd.read_csv(tmp_path / "runs_rmb.csv")) == 2

        lanjutan = runner.run_batch("rmb", grid, batch_id="percobaan_lanjutan")
        assert len(lanjutan) == 1
        assert lanjutan.loc[0, "epochs"] == 3
        assert len(pd.read_csv(tmp_path / "runs_rmb.csv")) == 3

    def test_batch_tanpa_konfigurasi_baru_mengembalikan_kosong(self, runner) -> None:
        grid = [{"config": {"epochs": 1}}]
        runner.run_batch("rmb", grid)
        assert runner.run_batch("rmb", grid).empty

    def test_resume_false_menjalankan_ulang(self, runner, tmp_path) -> None:
        """Pengukuran ulang yang disengaja tetap mungkin."""
        grid = [{"config": {"epochs": 1}}]
        runner.run_batch("rmb", grid)
        ulang = runner.run_batch("rmb", grid, resume=False)
        assert len(ulang) == 1
        assert len(pd.read_csv(tmp_path / "runs_rmb.csv")) == 2

    def test_penomoran_run_berlanjut_setelah_terputus(self, runner) -> None:
        runner.run_batch("rmb", [{"config": {"epochs": e}} for e in (1, 2)])
        lanjutan = runner.run_batch("rmb", [{"config": {"epochs": e}} for e in (1, 2, 3)])
        assert lanjutan.loc[0, "run_id"] == 3


@pytest.fixture
def tiga_head(runner):
    """Tiga head RM-b; run #1 linear, run #2 dan #3 MLP."""
    runner.run_batch(
        "rmb",
        [
            {"config": {"epochs": 2}},
            {"config": {"head_arch": "mlp", "hidden_dim": 16, "epochs": 2}},
            {"config": {"head_arch": "mlp", "hidden_dim": 32, "epochs": 3}},
        ],
    )
    return runner


class TestHeadRMBTersimpan:
    def test_setiap_run_menyimpan_state_head_bukan_hanya_juara(self, runner, tmp_path) -> None:
        runner.run_batch("rmb", [{"config": {"epochs": 1}}, {"config": {"epochs": 2}}])
        for run_id in (1, 2):
            assert (tmp_path / "checkpoints" / "rmb_heads" / f"run_{run_id}.pt").exists()

    def test_isi_state_head_sama_dengan_checkpoint_juara(self, runner, tmp_path) -> None:
        runner.run("rmb", {"epochs": 2})
        heads = torch.load(
            tmp_path / "checkpoints" / "rmb_heads" / "run_1.pt", map_location="cpu", weights_only=True
        )
        champion = torch.load(
            tmp_path / "checkpoints" / "rmb_best.pt", map_location="cpu", weights_only=True
        )
        assert heads["run_id"] == champion["run_id"] == 1
        for key, value in champion["head_state"].items():
            torch.testing.assert_close(heads["head_state"][key], value)

    def test_pemulihan_melatih_ulang_hanya_head_yang_belum_ada(self, runner, tmp_path) -> None:
        runner.run_batch("rmb", [{"config": {"epochs": 2}}, {"config": {"epochs": 2, "lr": 1e-3}}])
        (tmp_path / "checkpoints" / "rmb_heads" / "run_2.pt").unlink()

        assert runner.restore_rmb_heads() == [2]
        assert (tmp_path / "checkpoints" / "rmb_heads" / "run_2.pt").exists()
        assert runner.restore_rmb_heads() == []

    def test_pemulihan_mereproduksi_head_pada_mesin_yang_sama(self, runner, tmp_path) -> None:
        row = runner.run("rmb", {"epochs": 3})
        (tmp_path / "checkpoints" / "rmb_heads" / "run_1.pt").unlink()
        runner.restore_rmb_heads()

        restored = runner._load_rmb_head(1)
        with torch.no_grad():
            logits = restored(torch.tensor(runner.features.embeddings["val"]))
        from src.services.evaluation import ClassificationEvaluator

        metrics = ClassificationEvaluator().metrics(
            runner.features.labels["val"], logits.argmax(1).numpy()
        )
        assert metrics["f1_macro"] == pytest.approx(row["val_f1_macro"], abs=1e-5)

    def test_pemulihan_tanpa_riwayat_rmb_ditolak(self, runner) -> None:
        with pytest.raises(RuntimeError, match="runs_rmb.csv kosong"):
            runner.restore_rmb_heads()


class TestRMCSeluruhHead:
    """Satu grid RM-c: seluruh head RM-b x (alpha, k) dengan fusi linear."""

    def test_tanpa_rmb_run_id_memakai_dan_mencatat_juara_rmb(self, tiga_head) -> None:
        champion = tiga_head.best.get("rmb")["run_id"]
        row = tiga_head.run("rmc", {"alpha": 0.2, "k": 5})
        assert row["rmb_run_id"] == champion

    def test_rmb_run_id_memilih_head_yang_dipakai(self, tiga_head) -> None:
        linear = tiga_head.run("rmc", {"rmb_run_id": 1, "alpha": 0.2})
        mlp = tiga_head.run("rmc", {"rmb_run_id": 3, "alpha": 0.2})
        assert (linear["head_arch_rmb"], mlp["head_arch_rmb"]) == ("linear", "mlp")
        assert mlp["hidden_dim_rmb"] == 32

    def test_alpha_nol_mereproduksi_f1_head_rmb(self, tiga_head) -> None:
        row = tiga_head.run("rmc", {"rmb_run_id": 2, "alpha": 0.0})
        assert row["val_f1_macro"] == pytest.approx(row["val_f1_rmb"], abs=1e-6)

    def test_biaya_head_yang_dipakai_dicatat(self, tiga_head, tmp_path) -> None:
        rmb = pd.read_csv(tmp_path / "runs_rmb.csv").set_index("run_id")
        row = tiga_head.run("rmc", {"rmb_run_id": 3, "alpha": 0.2})
        assert row["head_trainable_params"] == rmb.loc[3, "trainable_params"]
        assert row["head_train_time_s"] == pytest.approx(rmb.loc[3, "train_time_s"])
        assert (row["trainable_params"], row["train_time_s"]) == (0, 0.0)

    def test_head_tak_dikenal_diisolasi_sebagai_error_batch(self, tiga_head, tmp_path) -> None:
        frame = tiga_head.run_batch(
            "rmc",
            [{"config": {"rmb_run_id": 9, "alpha": 0.2}}, {"config": {"rmb_run_id": 1, "alpha": 0.2}}],
        )
        assert len(frame) == 1
        assert len(pd.read_csv(tmp_path / "runs_rmc_errors.csv")) == 1

    def test_grid_seluruh_head_dijalankan_dan_resume(self, tiga_head, tmp_path) -> None:
        grid = [
            {"config": {"rmb_run_id": head, "alpha": alpha, "k": 3}}
            for head in (1, 2, 3)
            for alpha in (0.0, 0.5)
        ]
        assert len(tiga_head.run_batch("rmc", grid)) == 6
        assert tiga_head.run_batch("rmc", grid).empty
        assert sorted(pd.read_csv(tmp_path / "runs_rmc.csv")["rmb_run_id"].unique()) == [1, 2, 3]

    def test_rmb_run_id_kosong_dan_juara_eksplisit_dianggap_sama(self, tiga_head) -> None:
        champion = tiga_head.best.get("rmb")["run_id"]
        assert tiga_head.config_signature("rmc", {"alpha": 0.2}) == tiga_head.config_signature(
            "rmc", {"alpha": 0.2, "rmb_run_id": champion}
        )

    def test_juara_rmc_memilih_f1_tertinggi_lintas_head(self, tiga_head, tmp_path) -> None:
        tiga_head.run_batch(
            "rmc",
            [{"config": {"rmb_run_id": head, "alpha": 0.3, "k": 3}} for head in (1, 2, 3)],
        )
        runs = pd.read_csv(tmp_path / "runs_rmc.csv")
        best = tiga_head.best.get("rmc")
        assert best["val_f1_macro"] == pytest.approx(runs["val_f1_macro"].max())
        assert best["config"]["rmb_run_id"] == int(runs.loc[runs["val_f1_macro"].idxmax(), "rmb_run_id"])

    def test_split_test_tidak_dibuka(self, tiga_head, tmp_path) -> None:
        tiga_head.run("rmc", {"rmb_run_id": 2, "alpha": 0.2})
        columns = pd.read_csv(tmp_path / "runs_rmc.csv").columns
        assert not any(column.startswith("test_") for column in columns)


class TestPrediktorRMC:
    def test_juara_membawa_head_dan_fusi_linear_yang_dipakai(self, tiga_head, tmp_path) -> None:
        tiga_head.run("rmc", {"rmb_run_id": 3, "alpha": 0.2, "k": 5})

        checkpoint = torch.load(tmp_path / "checkpoints" / "rmc_best.pt", map_location="cpu", weights_only=True)
        head, config = tiga_head._load_rmc_predictor()

        assert {"head_state", "head_config", "hidden_size"} <= set(checkpoint)
        assert config == FusionFormulaConfig("linear", alpha=0.2, k=5, weighting="similarity")
        expected = tiga_head._load_rmb_head(3)
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(head.state_dict()[key], value)


class TestBiayaEfektifRMC:
    def test_juara_dibebani_biaya_head_yang_dipakainya(self, tiga_head, tmp_path) -> None:
        tiga_head.run("rmc", {"rmb_run_id": 2, "alpha": 0.2, "k": 5})
        rmb = pd.read_csv(tmp_path / "runs_rmb.csv").set_index("run_id")

        params, time_s = tiga_head._effective_cost("rmc")

        assert params == rmb.loc[2, "trainable_params"] > 0
        assert time_s == pytest.approx(rmb.loc[2, "train_time_s"])

    def test_catatan_lama_tanpa_biaya_head_memakai_juara_rmb(self, runner) -> None:
        """best.json dari kampanye sebelum perubahan ini mencatat RM-c sebagai nol."""
        row = runner.run("rmb", {"epochs": 2})
        runner.best.data["rmc"] = {"run_id": 1, "val_f1_macro": 0.9, "trainable_params": 0, "train_time_s": 0.0}

        assert runner._effective_cost("rmc")[0] == row["trainable_params"]

    def test_skenario_lain_memakai_catatan_juaranya_sendiri(self, runner) -> None:
        row = runner.run("rmb", {"epochs": 2})
        assert runner._effective_cost("rmb") == (row["trainable_params"], row["train_time_s"])


class TestPemulihanCheckpoint:
    """Checkpoint tidak ikut git tetapi best.json ikut; di clone baru keduanya tidak selaras."""

    @staticmethod
    def hapus_checkpoint(tmp_path) -> None:
        import shutil

        shutil.rmtree(tmp_path / "checkpoints")

    def test_pesan_error_menunjuk_ke_pemulihan(self, runner) -> None:
        with pytest.raises(FileNotFoundError, match="restore_checkpoints"):
            runner._load_checkpoint("rmb_best.pt")

    def test_menjalankan_ulang_rmb_tidak_membuat_juara_kembali(self, runner, tmp_path) -> None:
        """Alasan pemulihan dibutuhkan: F1 yang sama tidak dipromosikan, jadi tidak disimpan."""
        runner.run("rmb", {"epochs": 2})
        (tmp_path / "checkpoints" / "rmb_best.pt").unlink()

        runner.run("rmb", {"epochs": 2})

        assert not (tmp_path / "checkpoints" / "rmb_best.pt").exists()

    def test_juara_rmb_dibangun_kembali_dengan_bobot_yang_sama(self, runner, tmp_path) -> None:
        runner.run("rmb", {"epochs": 3})
        before = torch.load(tmp_path / "checkpoints" / "rmb_best.pt", map_location="cpu", weights_only=True)
        self.hapus_checkpoint(tmp_path)

        restored = runner.restore_checkpoints()

        assert restored["rmb_best"] is True
        after = torch.load(tmp_path / "checkpoints" / "rmb_best.pt", map_location="cpu", weights_only=True)
        assert after["run_id"] == before["run_id"]
        for key, value in before["head_state"].items():
            torch.testing.assert_close(after["head_state"][key], value)

    def test_head_yang_dimuat_setelah_pemulihan_dapat_dipakai(self, runner, tmp_path) -> None:
        runner.run("rmb", {"epochs": 2})
        self.hapus_checkpoint(tmp_path)
        runner.restore_checkpoints()

        head, config = runner._load_best_head()

        assert config["epochs"] == 2
        assert head(torch.tensor(runner.features.embeddings["val"])).shape[1] == 2

    def test_riwayat_dan_best_json_tidak_berubah(self, runner, tmp_path) -> None:
        runner.run("rmb", {"epochs": 2})
        runner.run("rmc", {"alpha": 0.2, "k": 5})
        history = {name: (tmp_path / name).read_text(encoding="utf-8") for name in ("runs_rmb.csv", "runs_rmc.csv", "best.json")}
        self.hapus_checkpoint(tmp_path)

        runner.restore_checkpoints()

        for name, content in history.items():
            assert (tmp_path / name).read_text(encoding="utf-8") == content

    def test_checkpoint_yang_sudah_ada_tidak_disentuh(self, runner, tmp_path) -> None:
        runner.run("rmb", {"epochs": 2})
        runner.run("rmc", {"alpha": 0.2, "k": 5})

        restored = runner.restore_checkpoints()

        assert restored["rmb_heads"] == []
        assert restored["rmb_best"] is False
        assert restored["rmc_best"] is False

    def test_juara_rmc_standar_dibangun_kembali_dari_konfigurasi_tercatat(self, runner, tmp_path) -> None:
        runner.run("rmb", {"epochs": 2})
        runner.run("rmc", {"alpha": 0.2, "k": 5})
        (tmp_path / "checkpoints" / "rmc_best.pt").unlink()

        assert runner.restore_checkpoints()["rmc_best"] is True

        head, config = runner._load_rmc_predictor()
        assert (config.formula, config.alpha, config.k) == ("linear", 0.2, 5)

    def test_juara_rmc_kembali_membawa_head_yang_dipakainya(self, tiga_head, tmp_path) -> None:
        tiga_head.run("rmc", {"rmb_run_id": 3, "alpha": 0.2, "k": 5})
        self.hapus_checkpoint(tmp_path)

        tiga_head.restore_checkpoints()

        head, config = tiga_head._load_rmc_predictor()
        assert (config.alpha, config.k) == (0.2, 5)
        expected = tiga_head._load_rmb_head(3)
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(head.state_dict()[key], value)

    def test_rma_tidak_dilatih_ulang_kecuali_diminta(self, runner, tmp_path, monkeypatch) -> None:
        runner.run("rmb", {"epochs": 1})

        def tidak_boleh_dipanggil(*args, **kwargs):
            raise AssertionError("RM-a tidak boleh dilatih ulang tanpa include_rma")

        monkeypatch.setattr("src.services.campaign.RMATrainer", tidak_boleh_dipanggil)
        assert runner.restore_checkpoints()["rma_best"] is False

    def test_rma_tanpa_juara_tercatat_tidak_dilatih(self, runner, monkeypatch) -> None:
        runner.run("rmb", {"epochs": 1})
        monkeypatch.setattr("src.services.campaign.RMATrainer", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
        assert runner.restore_checkpoints(include_rma=True)["rma_best"] is False

    def test_rma_dilatih_ulang_dari_konfigurasi_juara_bila_diminta(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        from types import SimpleNamespace

        from src.models.schemas import RMAConfig

        class PelatihPalsu:
            def __init__(self, data) -> None:
                pass

            def train(self, config):
                return SimpleNamespace(
                    best_metrics={"f1_macro": 0.98}, best_state={"w": torch.zeros(2)}
                ), None

        monkeypatch.setattr("src.services.campaign.RMATrainer", PelatihPalsu)
        runner.run("rmb", {"epochs": 1})
        runner.best.data["rma"] = {
            "run_id": 7, "config": RMAConfig().model_dump(), "val_f1_macro": 0.98,
        }

        assert runner.restore_checkpoints(include_rma=True)["rma_best"] is True

        payload = torch.load(tmp_path / "checkpoints" / "rma_best.pt", map_location="cpu", weights_only=True)
        assert payload["run_id"] == 7
        assert "w" in payload["model_state"]

    def test_tanpa_riwayat_rmb_ditolak(self, runner) -> None:
        with pytest.raises(RuntimeError, match="runs_rmb.csv kosong"):
            runner.restore_checkpoints()


def bootstrap_palsu(observed: float, ci_low: float):
    def palsu(*args, **kwargs):
        return {"observed_delta_pp": observed, "mean_delta_pp": observed, "ci_low_pp": ci_low,
                "ci_high_pp": observed + 1, "n_boot": 10_000, "seed": 42, "n_samples": 50}
    return palsu


@pytest.fixture
def grid_rmc(tiga_head):
    """Grid kecil di ketiga head; head resmi RM-b adalah juara RM-b."""
    tiga_head.run_batch(
        "rmc",
        [{"config": {"rmb_run_id": head, "alpha": alpha, "k": 3}}
         for head in (1, 2, 3) for alpha in (0.0, 0.3, 0.6)],
    )
    return tiga_head


class TestPutusanJuaraRMC:
    def test_default_adalah_terbaik_di_head_rmb_resmi(self, grid_rmc, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.services.campaign.paired_bootstrap_f1", bootstrap_palsu(0.1, -0.5))
        decision = grid_rmc.decide_rmc_champion()

        official = grid_rmc.best.get("rmb")["run_id"]
        runs = pd.read_csv(tmp_path / "runs_rmc.csv")
        on_official = runs[runs["rmb_run_id"] == official]
        assert decision["default"]["rmb_run_id"] == official
        assert decision["default"]["val_f1_macro"] == pytest.approx(on_official["val_f1_macro"].max())
        champion = grid_rmc.best.get("rmc")
        if not decision["challenger_is_default"]:
            assert decision["winner"] == "default"
        assert champion["config"]["rmb_run_id"] == official
        assert champion["head_is_official_rmb"] is True
        assert (tmp_path / "rmc_champion_decision.json").exists()

    def test_penantang_menang_membawa_head_dan_biayanya_sendiri(
        self, grid_rmc, tmp_path, monkeypatch
    ) -> None:
        official = grid_rmc.best.get("rmb")["run_id"]
        runs = pd.read_csv(tmp_path / "runs_rmc.csv")
        lain = runs[runs["rmb_run_id"] != official].sort_values("val_f1_macro").iloc[0]
        runs.loc[runs["run_id"] == lain["run_id"], "val_f1_macro"] = 0.999
        runs.to_csv(tmp_path / "runs_rmc.csv", index=False)
        monkeypatch.setattr("src.services.campaign.paired_bootstrap_f1", bootstrap_palsu(0.5, 0.1))

        decision = grid_rmc.decide_rmc_champion()

        champion = grid_rmc.best.get("rmc")
        assert decision["winner"] == "penantang"
        assert champion["head_is_official_rmb"] is False
        assert champion["config"]["rmb_run_id"] == int(lain["rmb_run_id"])
        rmb = pd.read_csv(tmp_path / "runs_rmb.csv").set_index("run_id").loc[int(lain["rmb_run_id"])]
        assert grid_rmc._effective_cost("rmc") == (rmb["trainable_params"], pytest.approx(rmb["train_time_s"]))
        assert "BUKAN warisan" in decision["cost_note"]
        head, _ = grid_rmc._load_rmc_predictor()
        expected = grid_rmc._load_rmb_head(int(lain["rmb_run_id"]))
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(head.state_dict()[key], value)

    def test_selisih_di_bawah_ambang_tetap_default(self, grid_rmc, tmp_path, monkeypatch) -> None:
        runs = pd.read_csv(tmp_path / "runs_rmc.csv")
        official = grid_rmc.best.get("rmb")["run_id"]
        runs.loc[runs["rmb_run_id"] != official, "val_f1_macro"] += 0.5
        runs.to_csv(tmp_path / "runs_rmc.csv", index=False)
        monkeypatch.setattr("src.services.campaign.paired_bootstrap_f1", bootstrap_palsu(0.14, 0.05))

        assert grid_rmc.decide_rmc_champion()["winner"] == "default"

    def test_run_susulan_tidak_mengubah_putusan(self, grid_rmc) -> None:
        grid_rmc.decide_rmc_champion()
        before = dict(grid_rmc.best.get("rmc"))
        grid_rmc.run("rmc", {"rmb_run_id": 2, "alpha": 0.9, "k": 1})
        assert grid_rmc.best.get("rmc") == before

    def test_bootstrap_sungguhan_tercatat(self, grid_rmc, tmp_path) -> None:
        runs = pd.read_csv(tmp_path / "runs_rmc.csv")
        official = grid_rmc.best.get("rmb")["run_id"]
        runs.loc[runs["rmb_run_id"] != official, "val_f1_macro"] += 0.5
        runs.to_csv(tmp_path / "runs_rmc.csv", index=False)

        decision = grid_rmc.decide_rmc_champion()

        assert decision["bootstrap"]["n_boot"] == 10_000
        assert decision["bootstrap"]["ci_low_pp"] <= decision["bootstrap"]["ci_high_pp"]

    def test_tanpa_riwayat_rmc_ditolak(self, tiga_head) -> None:
        with pytest.raises(RuntimeError, match="riwayat RM-c"):
            tiga_head.decide_rmc_champion()
