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


EXPLORATION_GRID = [
    FusionFormulaConfig("linear", alpha=0.0, k=1),
    FusionFormulaConfig("linear", alpha=0.3, k=5),
    FusionFormulaConfig("rumus1", k=5),
    FusionFormulaConfig("rumus2", k=5),
    FusionFormulaConfig("rumus3", k=5),
    FusionFormulaConfig("rumus4", alpha=0.3, k=5),
]


@pytest.fixture
def explored(runner):
    """Tiga head RM-b, satu run RM-c standar, lalu eksplorasi di atas ketiga head."""
    runner.run_batch(
        "rmb",
        [
            {"config": {"epochs": 2}},
            {"config": {"head_arch": "mlp", "hidden_dim": 16, "epochs": 2}},
            {"config": {"head_arch": "mlp", "hidden_dim": 32, "epochs": 3}},
        ],
    )
    runner.run("rmc", {"alpha": 0.2, "k": 5})
    return runner.explore_rmc(EXPLORATION_GRID)


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


class TestEksplorasiRMC:
    def test_seluruh_head_dan_konfigurasi_diuji(self, explored) -> None:
        assert len(explored["runs"]) == 3 * len(EXPLORATION_GRID)
        assert len(explored["per_head"]) == 3
        assert set(explored["per_formula"]["formula"]) == {
            "linear", "rumus1", "rumus2", "rumus3", "rumus4",
        }

    def test_hasil_ditulis_ke_folder_eksplorasi(self, explored, tmp_path) -> None:
        for name in ("runs", "per_head", "per_formula"):
            assert (tmp_path / "rmc_exploration" / f"rmc_exploration_{name}.csv").exists()

    def test_tidak_menyentuh_riwayat_rmc_standar_dan_juara(self, runner, tmp_path) -> None:
        runner.run("rmb", {"epochs": 2})
        runner.run("rmc", {"alpha": 0.2, "k": 5})
        history_before = (tmp_path / "runs_rmc.csv").read_text(encoding="utf-8")
        best_before = (tmp_path / "best.json").read_text(encoding="utf-8")

        runner.explore_rmc(EXPLORATION_GRID)

        assert (tmp_path / "runs_rmc.csv").read_text(encoding="utf-8") == history_before
        assert (tmp_path / "best.json").read_text(encoding="utf-8") == best_before

    def test_head_belum_tersimpan_diminta_dipulihkan(self, runner, tmp_path) -> None:
        runner.run("rmb", {"epochs": 1})
        (tmp_path / "checkpoints" / "rmb_heads" / "run_1.pt").unlink()
        with pytest.raises(FileNotFoundError, match="restore_rmb_heads"):
            runner.explore_rmc(EXPLORATION_GRID)

    def test_tanpa_riwayat_rmb_ditolak(self, runner) -> None:
        with pytest.raises(RuntimeError, match="runs_rmb.csv kosong"):
            runner.explore_rmc(EXPLORATION_GRID)

    def test_split_test_tidak_dibuka(self, explored) -> None:
        assert not any(column.startswith("test_") for column in explored["runs"].columns)


class TestPutusanJuaraRMC:
    def test_tanpa_juara_standar_ditolak(self, runner, explored) -> None:
        (runner.out_dir / "best.json").write_text("{}", encoding="utf-8")
        runner.best.data = {}
        with pytest.raises(RuntimeError, match="juara RM-c standar belum ada"):
            runner.decide_rmc_champion(explored["runs"])

    def test_penantang_kalah_membiarkan_juara_standar(self, runner, explored, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.services.campaign.challenger_wins", lambda *args, **kwargs: False)
        best_before = (tmp_path / "best.json").read_text(encoding="utf-8")

        decision = runner.decide_rmc_champion(explored["runs"])

        assert decision["winner"] == "standar"
        assert (tmp_path / "best.json").read_text(encoding="utf-8") == best_before
        assert (tmp_path / "rmc_exploration" / "champion_decision.json").exists()

    def test_penantang_menang_mengganti_juara_dan_membawa_head_sendiri(
        self, runner, explored, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr("src.services.campaign.challenger_wins", lambda *args, **kwargs: True)

        decision = runner.decide_rmc_champion(explored["runs"])

        assert decision["winner"] == "eksplorasi"
        champion = runner.best.get("rmc")
        assert champion["source"] == "eksplorasi"
        assert {"rmb_run_id", "formula", "k", "weighting"} <= set(champion["config"])
        checkpoint = torch.load(tmp_path / "checkpoints" / "rmc_best.pt", map_location="cpu", weights_only=True)
        assert {"head_state", "head_config", "hidden_size"} <= set(checkpoint)

    def test_catatan_keputusan_memuat_selisih_dan_interval(self, runner, explored) -> None:
        decision = runner.decide_rmc_champion(explored["runs"])
        assert decision["winner"] in {"standar", "eksplorasi"}
        low, high = decision["ci95_pp"]
        assert low <= high
        assert decision["tie_threshold_pp"] == pytest.approx(0.15)

    def test_run_standar_susulan_tidak_menimpa_putusan_eksplorasi(
        self, runner, explored, monkeypatch
    ) -> None:
        monkeypatch.setattr("src.services.campaign.challenger_wins", lambda *args, **kwargs: True)
        runner.decide_rmc_champion(explored["runs"])

        runner.run("rmc", {"alpha": 0.9, "k": 3})

        assert runner.best.get("rmc")["source"] == "eksplorasi"


class TestPrediktorRMC:
    def test_juara_standar_memakai_head_juara_rmb_dan_fusi_linear(self, runner) -> None:
        runner.run("rmb", {"epochs": 2})
        runner.run("rmc", {"alpha": 0.2, "k": 5})

        head, config = runner._load_rmc_predictor()

        assert config == FusionFormulaConfig("linear", alpha=0.2, k=5, weighting="similarity")
        expected, _ = runner._load_best_head()
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(head.state_dict()[key], value)

    def test_juara_eksplorasi_memuat_head_dan_rumusnya_sendiri(
        self, runner, explored, monkeypatch
    ) -> None:
        monkeypatch.setattr("src.services.campaign.challenger_wins", lambda *args, **kwargs: True)
        decision = runner.decide_rmc_champion(explored["runs"])

        head, config = runner._load_rmc_predictor()

        assert config.formula == decision["challenger"]["formula"]
        assert config.k == decision["challenger"]["k"]
        expected = runner._load_rmb_head(decision["challenger"]["rmb_run_id"])
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(head.state_dict()[key], value)


class TestBiayaEfektifRMC:
    def test_juara_standar_dibebani_biaya_head_rmb(self, runner) -> None:
        row = runner.run("rmb", {"epochs": 2})
        runner.run("rmc", {"alpha": 0.2, "k": 5})

        params, time_s = runner._effective_cost("rmc")

        assert params == row["trainable_params"] > 0
        assert time_s == pytest.approx(row["train_time_s"])

    def test_catatan_lama_tanpa_biaya_head_memakai_juara_rmb(self, runner) -> None:
        """best.json dari kampanye sebelum perubahan ini mencatat RM-c sebagai nol."""
        row = runner.run("rmb", {"epochs": 2})
        runner.best.data["rmc"] = {"run_id": 1, "val_f1_macro": 0.9, "trainable_params": 0, "train_time_s": 0.0}

        assert runner._effective_cost("rmc")[0] == row["trainable_params"]

    def test_juara_eksplorasi_dibebani_biaya_head_yang_dipakainya(
        self, runner, explored, monkeypatch
    ) -> None:
        monkeypatch.setattr("src.services.campaign.challenger_wins", lambda *args, **kwargs: True)
        decision = runner.decide_rmc_champion(explored["runs"])

        used = explored["runs"].query("rmb_run_id == @decision['challenger']['rmb_run_id']").iloc[0]
        assert runner._effective_cost("rmc") == (int(used["head_params"]), float(used["head_train_time_s"]))

    def test_skenario_lain_memakai_catatan_juaranya_sendiri(self, runner) -> None:
        row = runner.run("rmb", {"epochs": 2})
        assert runner._effective_cost("rmb") == (row["trainable_params"], row["train_time_s"])
