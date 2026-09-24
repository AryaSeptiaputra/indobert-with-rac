"""Test metrik klasifikasi, pengukur efisiensi, dan gate lingkungan satu-hardware (Aturan #1, Tabel 4.1)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.services import evaluation
from src.services.evaluation import (
    RUN_METRIC_KEYS,
    ClassificationEvaluator,
    EfficiencyProfiler,
    EnvironmentMismatchError,
    record_tuning_session,
    verify_final_session,
)


class TestClassificationEvaluator:
    def test_prediksi_sempurna(self) -> None:
        metrics = ClassificationEvaluator().metrics([0, 1, 0, 1], [0, 1, 0, 1])
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["f1_class1"] == pytest.approx(1.0)

    def test_akurasi_menyesatkan_pada_data_timpang(self) -> None:
        """Menebak semua nol pada data 90:10 memberi akurasi 0,9 tapi F1 judi nol.

        Inilah alasan metrik utama penelitian ini F1-macro, bukan accuracy.
        """
        y_true = [0] * 90 + [1] * 10
        metrics = ClassificationEvaluator().metrics(y_true, [0] * 100)
        assert metrics["accuracy"] == pytest.approx(0.9)
        assert metrics["f1_class1"] == pytest.approx(0.0)
        assert metrics["f1_macro"] < 0.5

    def test_kelas_tanpa_prediksi_tidak_membagi_nol(self) -> None:
        metrics = ClassificationEvaluator().metrics([0, 0, 1], [0, 0, 0])
        assert np.isfinite(metrics["precision_class1"])
        assert metrics["precision_class1"] == pytest.approx(0.0)

    def test_support_menghitung_jumlah_sebenarnya(self) -> None:
        metrics = ClassificationEvaluator().metrics([0, 0, 0, 1], [0, 0, 0, 1])
        assert metrics["support_class0"] == 3
        assert metrics["support_class1"] == 1

    def test_panjang_berbeda_ditolak(self) -> None:
        with pytest.raises(ValueError, match="panjang"):
            ClassificationEvaluator().metrics([0, 1], [0, 1, 0])

    def test_confusion_matrix_berurutan_tetap(self) -> None:
        matrix = ClassificationEvaluator().confusion([0, 0, 1, 1], [0, 1, 0, 1])
        np.testing.assert_array_equal(matrix, [[1, 1], [1, 1]])

    def test_confusion_matrix_tetap_dua_kali_dua_walau_satu_kelas(self) -> None:
        matrix = ClassificationEvaluator().confusion([0, 0], [0, 0])
        assert matrix.shape == (2, 2)

    def test_average_precision_lebih_baik_dari_acak(self, rng) -> None:
        y_true = rng.integers(0, 2, 200)
        skor_bagus = y_true * 0.7 + rng.normal(0, 0.1, 200)
        evaluator = ClassificationEvaluator()
        assert evaluator.average_precision(y_true, skor_bagus) > 0.9

    def test_run_metrics_menyaring_kolom_ringkas(self) -> None:
        evaluator = ClassificationEvaluator()
        metrics = evaluator.metrics([0, 1], [0, 1])
        ringkas = evaluator.run_metrics(metrics, prefix="test")
        assert set(ringkas) == {f"test_{key}" for key in RUN_METRIC_KEYS}


class TestEfficiencyProfiler:
    def test_menghitung_parameter_trainable(self) -> None:
        model = torch.nn.Linear(10, 2)
        counts = EfficiencyProfiler().count_parameters(model)
        assert counts["total_params"] == 22
        assert counts["trainable_params"] == 22
        assert counts["trainable_pct"] == pytest.approx(100.0)

    def test_parameter_beku_tidak_dihitung_trainable(self) -> None:
        """Inti klaim efisiensi RM-b: encoder beku tidak masuk hitungan."""
        model = torch.nn.Sequential(torch.nn.Linear(10, 8), torch.nn.Linear(8, 2))
        model[0].requires_grad_(False)
        counts = EfficiencyProfiler().count_parameters(model)
        assert counts["total_params"] == 88 + 18
        assert counts["trainable_params"] == 18
        assert counts["trainable_pct"] < 20

    def test_model_tanpa_parameter_tidak_membagi_nol(self) -> None:
        counts = EfficiencyProfiler().count_parameters(torch.nn.ReLU())
        assert counts["trainable_pct"] == 0.0

    def test_memori_puncak_nol_di_cpu(self) -> None:
        profiler = EfficiencyProfiler()
        profiler.reset_peak_memory()
        if not torch.cuda.is_available():
            assert profiler.peak_gpu_memory_mb() == 0.0

    def test_latency_positif_dan_menghormati_jumlah_run(self) -> None:
        calls = {"n": 0}

        def predict() -> None:
            calls["n"] += 1

        profiler = EfficiencyProfiler(n_warmup=2, n_runs=5)
        latency = profiler.measure_latency(predict)
        assert latency >= 0.0
        assert calls["n"] == 7

    def test_kegagalan_saat_pemanasan_dilempar(self) -> None:
        def predict() -> None:
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="out of memory"):
            EfficiencyProfiler(n_warmup=1, n_runs=1).measure_latency(predict)

    def test_protokol_pengukuran_sama_untuk_semua_skenario(self) -> None:
        """Latency RM-a/b/c hanya sebanding bila diukur dengan protokol identik."""
        first, second = EfficiencyProfiler(), EfficiencyProfiler()
        assert (first.n_warmup, first.n_runs) == (second.n_warmup, second.n_runs)


class TestBootstrapBerpasangan:
    def test_prediksi_identik_selisih_nol(self) -> None:
        from src.services.evaluation import paired_bootstrap_f1

        y = np.array([0, 1] * 50)
        hasil = paired_bootstrap_f1(y, y, y, n_boot=500)
        assert hasil["observed_delta_pp"] == hasil["ci_low_pp"] == hasil["ci_high_pp"] == 0.0

    def test_penantang_lebih_baik_selisih_positif(self) -> None:
        from src.services.evaluation import paired_bootstrap_f1

        y = np.array([0, 1] * 100)
        buruk = y.copy()
        buruk[:40] = 1 - buruk[:40]
        hasil = paired_bootstrap_f1(y, y, buruk, n_boot=2_000)
        assert hasil["observed_delta_pp"] > 0
        assert hasil["ci_low_pp"] > 0
        assert hasil["n_boot"] == 2_000 and hasil["n_samples"] == 200

    def test_seed_tetap_hasil_tetap(self) -> None:
        from src.services.evaluation import paired_bootstrap_f1

        rng = np.random.default_rng(1)
        y, a, b = (rng.integers(0, 2, 300) for _ in range(3))
        assert paired_bootstrap_f1(y, a, b, n_boot=300) == paired_bootstrap_f1(y, a, b, n_boot=300)


class TestLatencyBertahap:
    def test_keluaran_tahap_diteruskan_dan_urutan_dipertahankan(self) -> None:
        from src.services.evaluation import EfficiencyProfiler

        seen = []
        stages = (
            ("encoder", lambda _: 2),
            ("classification head", lambda value: seen.append(value) or value * 3),
            ("fusi", lambda value: seen.append(value)),
        )
        result = EfficiencyProfiler(n_warmup=1, n_runs=4).measure_stages(stages)
        assert list(result) == ["encoder", "classification head", "fusi"]
        assert all(value >= 0 for value in result.values())
        assert seen[:2] == [2, 6]
        assert len(seen) == 2 * (1 + 4)


LINGKUNGAN = {
    "gpu": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576, "driver": "550.54",
    "cuda_version": "12.4", "cpu": "AMD EPYC", "cpu_logical_cores": 32, "ram_total_gb": 125.7,
    "os": "Linux", "python": "3.13.1", "torch": "2.12.1", "transformers": "5.12.1",
    "faiss": "1.14.3", "seed": 42,
}


@pytest.fixture
def mesin(monkeypatch):
    state = {"env": dict(LINGKUNGAN), "boot": "2026-09-24 01:00:00"}
    monkeypatch.setattr(evaluation, "collect_environment", lambda: dict(state["env"]))
    monkeypatch.setattr(
        evaluation, "current_session",
        lambda: {"recorded_at": "2026-09-24 02:00:00", "boot_time": state["boot"]},
    )
    return state


class TestSesiTuning:
    def test_lingkungan_dan_sesi_tercatat(self, mesin, tmp_path) -> None:
        payload = record_tuning_session(tmp_path / "hardware.json")
        assert payload["gpu"] == LINGKUNGAN["gpu"]
        assert payload["faiss"] == "1.14.3" and payload["seed"] == 42
        assert len(payload["tuning_sessions"]) == 1

    def test_sesi_baru_ditumpuk_bukan_ditimpa(self, mesin, tmp_path) -> None:
        record_tuning_session(tmp_path / "hardware.json")
        record_tuning_session(tmp_path / "hardware.json")
        mesin["boot"] = "2026-09-25 08:00:00"
        payload = record_tuning_session(tmp_path / "hardware.json")
        assert [s["boot_time"] for s in payload["tuning_sessions"]] == [
            "2026-09-24 01:00:00", "2026-09-25 08:00:00",
        ]

    def test_lingkungan_lain_ditolak(self, mesin, tmp_path) -> None:
        record_tuning_session(tmp_path / "hardware.json")
        mesin["env"]["gpu"] = "NVIDIA GeForce RTX 3050 Laptop GPU"
        with pytest.raises(EnvironmentMismatchError, match="gpu"):
            record_tuning_session(tmp_path / "hardware.json")


class TestGateFinal:
    def test_lingkungan_identik_sesi_sama(self, mesin, tmp_path) -> None:
        record_tuning_session(tmp_path / "hardware.json")
        payload = verify_final_session(tmp_path / "hardware.json")
        assert payload["final_session"]["same_boot_as_tuning"] is True

    def test_boot_berbeda_hanya_peringatan_dan_tercatat(self, mesin, tmp_path) -> None:
        record_tuning_session(tmp_path / "hardware.json")
        mesin["boot"] = "2026-09-26 09:00:00"
        payload = verify_final_session(tmp_path / "hardware.json")
        saved = json.loads((tmp_path / "hardware.json").read_text(encoding="utf-8"))
        assert payload["final_session"]["same_boot_as_tuning"] is False
        assert saved["final_session"]["boot_time"] == "2026-09-26 09:00:00"
        assert saved["tuning_sessions"][0]["boot_time"] == "2026-09-24 01:00:00"

    @pytest.mark.parametrize("key", ["gpu", "vram_total_mb", "driver", "cuda_version", "torch", "transformers", "faiss"])
    def test_kunci_ketat_berbeda_menghentikan(self, mesin, tmp_path, key) -> None:
        record_tuning_session(tmp_path / "hardware.json")
        mesin["env"][key] = "lain"
        with pytest.raises(EnvironmentMismatchError, match=key):
            verify_final_session(tmp_path / "hardware.json")

    def test_tanpa_catatan_tuning_ditolak(self, mesin, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="03a"):
            verify_final_session(tmp_path / "hardware.json")


def test_lingkungan_sungguhan_lengkap() -> None:
    env = evaluation.collect_environment()
    assert set(env) >= {"gpu", "cpu", "ram_total_gb", "os", "python", "torch", "faiss", "seed"}
