"""Test engine training RM-b dan evaluator RM-c di atas fitur sintetis."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.heads import build_head
from src.models.schemas import RMBConfig, RMCConfig
from src.services.training import RMBTrainer, RMCEvaluator, history_row


class TestHistoryRow:
    def test_memuat_kolom_yang_dipakai_deteksi_overfit(self) -> None:
        row = history_row(
            2,
            0.1234567,
            {
                "f1_macro": 0.9,
                "accuracy": 0.95,
                "f1_class1": 0.8,
                "precision_class1": 0.82,
                "recall_class1": 0.78,
            },
        )
        assert row["epoch"] == 2
        assert row["train_loss"] == pytest.approx(0.123457)
        assert set(row) == {
            "epoch", "train_loss", "val_f1_macro", "val_acc",
            "val_f1_judi", "val_precision_judi", "val_recall_judi",
        }


class TestRMBTrainer:
    def test_menghasilkan_satu_baris_history_per_epoch(
        self, feature_set, class_weights, cpu_device
    ) -> None:
        result = RMBTrainer(feature_set, class_weights, cpu_device).train(
            RMBConfig(epochs=3)
        )
        assert len(result.history) == 3
        assert [row["epoch"] for row in result.history] == [1, 2, 3]

    def test_epoch_terbaik_berada_dalam_rentang(
        self, feature_set, class_weights, cpu_device
    ) -> None:
        result = RMBTrainer(feature_set, class_weights, cpu_device).train(
            RMBConfig(epochs=4)
        )
        assert 1 <= result.best_epoch <= 4

    def test_metrik_terbaik_sama_dengan_epoch_terbaik(
        self, feature_set, class_weights, cpu_device
    ) -> None:
        result = RMBTrainer(feature_set, class_weights, cpu_device).train(
            RMBConfig(epochs=4)
        )
        curve = [row["val_f1_macro"] for row in result.history]
        assert result.best_metrics["f1_macro"] == pytest.approx(max(curve))
        assert result.history[result.best_epoch - 1]["val_f1_macro"] == pytest.approx(
            max(curve)
        )

    def test_seed_sama_menghasilkan_hasil_identik(
        self, feature_set, class_weights, cpu_device
    ) -> None:
        """Tanpa ini, dua run dengan konfigurasi sama tidak bisa dibandingkan."""
        trainer = RMBTrainer(feature_set, class_weights, cpu_device)
        first = trainer.train(RMBConfig(epochs=3, seed=42))
        second = trainer.train(RMBConfig(epochs=3, seed=42))
        assert first.best_metrics["f1_macro"] == pytest.approx(
            second.best_metrics["f1_macro"]
        )
        assert [r["train_loss"] for r in first.history] == [
            r["train_loss"] for r in second.history
        ]

    def test_head_mlp_punya_parameter_jauh_lebih_banyak(
        self, feature_set, class_weights, cpu_device
    ) -> None:
        trainer = RMBTrainer(feature_set, class_weights, cpu_device)
        linear = trainer.train(RMBConfig(head_arch="linear", epochs=1))
        mlp = trainer.train(RMBConfig(head_arch="mlp", hidden_dim=256, epochs=1))
        assert mlp.trainable_params > linear.trainable_params * 10

    def test_parameter_head_linear_sesuai_rumus(
        self, feature_set, class_weights, cpu_device
    ) -> None:
        result = RMBTrainer(feature_set, class_weights, cpu_device).train(
            RMBConfig(head_arch="linear", epochs=1)
        )
        assert result.trainable_params == (feature_set.hidden_dim + 1) * 2

    def test_state_terbaik_bisa_dimuat_ulang(
        self, feature_set, class_weights, cpu_device
    ) -> None:
        config = RMBConfig(head_arch="mlp", hidden_dim=64, epochs=2)
        result = RMBTrainer(feature_set, class_weights, cpu_device).train(config)
        head = build_head(
            config.head_arch,
            hidden_size=feature_set.hidden_dim,
            dropout=config.dropout,
            hidden_dim=config.hidden_dim,
        )
        head.load_state_dict(result.best_state)

    def test_extras_memuat_latency_dan_hidden_size(
        self, feature_set, class_weights, cpu_device
    ) -> None:
        result = RMBTrainer(feature_set, class_weights, cpu_device).train(
            RMBConfig(epochs=1)
        )
        assert result.extras["hidden_size"] == feature_set.hidden_dim
        assert result.extras["infer_latency_ms"] >= 0.0

    def test_test_set_hanya_dibuka_bila_diminta(
        self, feature_set, class_weights, cpu_device
    ) -> None:
        trainer = RMBTrainer(feature_set, class_weights, cpu_device)
        assert "test_metrics" not in trainer.train(RMBConfig(epochs=1)).extras
        assert "test_metrics" in trainer.train(
            RMBConfig(epochs=1), eval_test=True
        ).extras

    def test_prediksi_test_sepanjang_split_test(
        self, feature_set, class_weights, cpu_device
    ) -> None:
        result = RMBTrainer(feature_set, class_weights, cpu_device).train(
            RMBConfig(epochs=1), eval_test=True
        )
        assert len(result.extras["test_pred"]) == len(feature_set.labels["test"])

    def test_waktu_latih_tercatat(self, feature_set, class_weights, cpu_device) -> None:
        result = RMBTrainer(feature_set, class_weights, cpu_device).train(
            RMBConfig(epochs=1)
        )
        assert result.train_time_s >= 0.0


class TestRMCEvaluator:
    @pytest.fixture
    def trained_head(self, feature_set, class_weights, cpu_device) -> torch.nn.Module:
        config = RMBConfig(head_arch="mlp", hidden_dim=64, epochs=3)
        result = RMBTrainer(feature_set, class_weights, cpu_device).train(config)
        head = build_head(
            config.head_arch,
            hidden_size=feature_set.hidden_dim,
            dropout=config.dropout,
            hidden_dim=config.hidden_dim,
        )
        head.load_state_dict(result.best_state)
        return head

    def test_indeks_hanya_dari_split_train(
        self, feature_set, trained_head, cpu_device
    ) -> None:
        """Indeks yang memuat val/test akan membuat retrieval menemukan jawabannya."""
        _, extras = RMCEvaluator(feature_set, trained_head, cpu_device).evaluate(
            RMCConfig(), split="val"
        )
        assert extras["index_vectors"] == len(feature_set.labels["train"])

    def test_tidak_melatih_apa_pun(self, feature_set, trained_head, cpu_device) -> None:
        params_before = [p.clone() for p in trained_head.parameters()]
        RMCEvaluator(feature_set, trained_head, cpu_device).evaluate(RMCConfig())
        for before, after in zip(params_before, trained_head.parameters()):
            torch.testing.assert_close(before, after)

    def test_alpha_nol_setara_head_murni(
        self, feature_set, trained_head, cpu_device
    ) -> None:
        evaluator = RMCEvaluator(feature_set, trained_head, cpu_device)
        metrics, _ = evaluator.evaluate(RMCConfig(alpha=0.0), split="val")

        with torch.no_grad():
            logits = trained_head(
                torch.tensor(feature_set.embeddings["val"], device=cpu_device)
            )
        expected = logits.argmax(1).cpu().numpy()

        from src.services.evaluation import ClassificationEvaluator

        head_only = ClassificationEvaluator().metrics(feature_set.labels["val"], expected)
        assert metrics["f1_macro"] == pytest.approx(head_only["f1_macro"])

    def test_prediksi_sepanjang_split_yang_dievaluasi(
        self, feature_set, trained_head, cpu_device
    ) -> None:
        _, extras = RMCEvaluator(feature_set, trained_head, cpu_device).evaluate(
            RMCConfig(), split="test"
        )
        assert len(extras["preds"]) == len(feature_set.labels["test"])
        assert len(extras["p_judi"]) == len(feature_set.labels["test"])

    def test_probabilitas_judi_dalam_rentang_sah(
        self, feature_set, trained_head, cpu_device
    ) -> None:
        _, extras = RMCEvaluator(feature_set, trained_head, cpu_device).evaluate(
            RMCConfig()
        )
        assert np.all(extras["p_judi"] >= 0.0)
        assert np.all(extras["p_judi"] <= 1.0)

    def test_k_melebihi_ukuran_train_ditolak(
        self, feature_set, trained_head, cpu_device
    ) -> None:
        evaluator = RMCEvaluator(feature_set, trained_head, cpu_device)
        oversized = len(feature_set.labels["train"]) + 1
        with pytest.raises(ValueError, match="melebihi"):
            evaluator.evaluate(RMCConfig(k=oversized))
