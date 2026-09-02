"""Test metrik klasifikasi dan pengukur efisiensi."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.services.evaluation import (
    RUN_METRIC_KEYS,
    ClassificationEvaluator,
    EfficiencyProfiler,
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
