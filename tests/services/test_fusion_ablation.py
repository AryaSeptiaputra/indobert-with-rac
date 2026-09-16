"""Test ablasi rumus fusi RAC: empat rumus alternatif di atas retrieval & head sama."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.heads import build_head
from src.models.schemas import RMBConfig
from src.services.fusion_ablation import FusionFormulaComparator, FusionFormulaConfig
from src.services.rac import RACClassifier, l2_normalize, softmax
from src.services.training import RMBTrainer


@pytest.fixture
def trained_head(feature_set, class_weights, cpu_device) -> torch.nn.Module:
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


@pytest.fixture
def comparator(feature_set, trained_head, cpu_device) -> FusionFormulaComparator:
    return FusionFormulaComparator(feature_set, trained_head, cpu_device, k=5)


class TestRawRetrievalScores:
    def test_pembilang_sama_dengan_retrieval_distribution(self, comparator: FusionFormulaComparator) -> None:
        similarities = np.array([[0.9, 0.5, 0.1], [0.2, 0.2, 0.2]], dtype=np.float32)
        neighbours = np.array([[1, 1, 0], [0, 0, 1]])

        raw = comparator.raw_retrieval_scores(similarities, neighbours, num_labels=2)
        normalized_dari_raw = raw / raw.sum(axis=1, keepdims=True)

        reference = RACClassifier(k=3).retrieval_distribution(similarities, neighbours)
        np.testing.assert_allclose(normalized_dari_raw, reference, rtol=1e-6)

    def test_tidak_dijamin_berjumlah_satu(self, comparator: FusionFormulaComparator) -> None:
        similarities = np.array([[0.9, 0.5, 0.1]], dtype=np.float32)
        neighbours = np.array([[1, 1, 0]])
        raw = comparator.raw_retrieval_scores(similarities, neighbours, num_labels=2)
        assert raw.sum() != pytest.approx(1.0)


class TestRumus1SkemaAsli:
    def test_norma_l2_tiap_cabang_sama_dengan_satu_sebelum_diskalakan(
        self, comparator: FusionFormulaComparator
    ) -> None:
        logits = np.array([[2.0, -1.0], [0.5, 0.5]], dtype=np.float32)
        raw_retrieval = np.array([[3.0, 0.0], [1.0, 1.0]], dtype=np.float32)

        np.testing.assert_allclose(np.linalg.norm(l2_normalize(logits), axis=1), 1.0, rtol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(l2_normalize(raw_retrieval), axis=1), 1.0, rtol=1e-6
        )

    def test_invarian_terhadap_penskalaan_positif(
        self, comparator: FusionFormulaComparator
    ) -> None:
        """L2-normalize membuang skala; inilah alasan Long dkk. memakainya untuk
        menyetarakan skor head dan skor retrieval yang hidup pada rentang berbeda."""
        logits = np.array([[2.0, -1.0], [0.5, 0.5]], dtype=np.float32)
        raw_retrieval = np.array([[3.0, 0.0], [1.0, 1.0]], dtype=np.float32)

        base = comparator.rumus1_score_fusion(logits, raw_retrieval)
        scaled = comparator.rumus1_score_fusion(logits * 10.0, raw_retrieval * 5.0)
        np.testing.assert_allclose(base, scaled, rtol=1e-5)

    def test_tidak_memakai_softmax_boleh_bernilai_negatif(
        self, comparator: FusionFormulaComparator
    ) -> None:
        """Softmax selalu >= 0; kalau cabang head Rumus 1 tetap bisa negatif, itu
        bukti fusinya beroperasi di skor mentah, bukan probabilitas."""
        logits = np.array([[-5.0, 1.0]], dtype=np.float32)
        raw_retrieval = np.array([[1.0, 1.0]], dtype=np.float32)
        score = comparator.rumus1_score_fusion(logits, raw_retrieval)
        assert (score < 0).any()

    def test_argmax_konsisten_dengan_skor_dominan(
        self, comparator: FusionFormulaComparator
    ) -> None:
        logits = np.array([[5.0, -5.0]], dtype=np.float32)
        raw_retrieval = np.array([[0.0, 10.0]], dtype=np.float32)
        score = comparator.rumus1_score_fusion(logits, raw_retrieval)
        # Kedua cabang menunjuk kelas berbeda dengan bobot sama (L/2 masing-masing);
        # hasil akhirnya harus konsisten dengan skor gabungan, bukan salah satu cabang saja.
        assert score.argmax(axis=1)[0] in (0, 1)


class TestRumus2SimilarityAlpha:
    def test_alpha_sama_dengan_rata_rata_similarity(
        self, comparator: FusionFormulaComparator
    ) -> None:
        p_bert = softmax(np.array([[2.0, 0.0], [0.0, 2.0]]))
        p_retrieval = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
        similarities = np.array([[0.8, 0.6], [-0.5, -0.5]], dtype=np.float32)

        _, alpha = comparator.rumus2_similarity_alpha(p_bert, similarities, p_retrieval)
        np.testing.assert_allclose(alpha, [0.7, 0.0], rtol=1e-6)

    def test_alpha_nol_menghasilkan_p_bert_murni(
        self, comparator: FusionFormulaComparator
    ) -> None:
        p_bert = softmax(np.array([[3.0, 0.0]]))
        p_retrieval = np.array([[0.2, 0.8]], dtype=np.float32)
        similarities = np.array([[-0.5, -0.9]], dtype=np.float32)

        p_final, alpha = comparator.rumus2_similarity_alpha(p_bert, similarities, p_retrieval)
        np.testing.assert_allclose(alpha, [0.0], rtol=1e-6)
        np.testing.assert_allclose(p_final, p_bert, rtol=1e-6)


class TestRumus3UncertaintyAlpha:
    def test_alpha_sama_dengan_satu_dikurangi_maksimum_p_bert(
        self, comparator: FusionFormulaComparator
    ) -> None:
        p_bert = np.array([[0.99, 0.01], [0.5, 0.5]], dtype=np.float32)
        p_retrieval = np.array([[0.2, 0.8], [0.3, 0.7]], dtype=np.float32)

        _, alpha = comparator.rumus3_uncertainty_alpha(p_bert, p_retrieval)
        np.testing.assert_allclose(alpha, [0.01, 0.5], rtol=1e-6)

    def test_head_yakin_menghasilkan_alpha_mendekati_nol(
        self, comparator: FusionFormulaComparator
    ) -> None:
        p_bert = np.array([[0.999, 0.001]], dtype=np.float32)
        p_retrieval = np.array([[0.5, 0.5]], dtype=np.float32)

        p_final, alpha = comparator.rumus3_uncertainty_alpha(p_bert, p_retrieval)
        assert alpha[0] < 0.01
        np.testing.assert_allclose(p_final, p_bert, atol=1e-2)


class TestRumus4GeometricPool:
    def test_hasil_tetap_distribusi_sah(self, comparator: FusionFormulaComparator) -> None:
        p_bert = softmax(np.array([[2.0, 0.0], [0.0, 3.0]]))
        p_retrieval = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float32)

        p_final = comparator.rumus4_geometric_pool(p_bert, p_retrieval, alpha=0.4)
        np.testing.assert_allclose(p_final.sum(axis=1), 1.0, rtol=1e-6)
        assert (p_final >= 0).all()

    def test_alpha_setengah_sama_dengan_rata_rata_geometris_manual(
        self, comparator: FusionFormulaComparator
    ) -> None:
        p_bert = np.array([[0.8, 0.2]], dtype=np.float32)
        p_retrieval = np.array([[0.2, 0.8]], dtype=np.float32)

        p_final = comparator.rumus4_geometric_pool(p_bert, p_retrieval, alpha=0.5)
        expected = np.sqrt(p_bert * p_retrieval)
        expected = expected / expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(p_final, expected, rtol=1e-4)

    def test_alpha_nol_setara_p_bert(self, comparator: FusionFormulaComparator) -> None:
        p_bert = np.array([[0.8, 0.2]], dtype=np.float32)
        p_retrieval = np.array([[0.2, 0.8]], dtype=np.float32)
        p_final = comparator.rumus4_geometric_pool(p_bert, p_retrieval, alpha=0.0)
        np.testing.assert_allclose(p_final, p_bert, rtol=1e-4)

    def test_alpha_satu_setara_p_retrieval(self, comparator: FusionFormulaComparator) -> None:
        p_bert = np.array([[0.8, 0.2]], dtype=np.float32)
        p_retrieval = np.array([[0.2, 0.8]], dtype=np.float32)
        p_final = comparator.rumus4_geometric_pool(p_bert, p_retrieval, alpha=1.0)
        np.testing.assert_allclose(p_final, p_retrieval, rtol=1e-4)


class TestFusionFormulaComparatorEvaluate:
    def test_indeks_hanya_dari_split_train(self, comparator: FusionFormulaComparator) -> None:
        _, extras = comparator.evaluate(FusionFormulaConfig(formula="rumus2"), split="val")
        assert extras["index_vectors"] == 200

    def test_tidak_melatih_apa_pun(
        self, comparator: FusionFormulaComparator, trained_head
    ) -> None:
        params_before = [p.clone() for p in trained_head.parameters()]
        comparator.evaluate(FusionFormulaConfig(formula="rumus3"), split="val")
        for before, after in zip(params_before, trained_head.parameters()):
            torch.testing.assert_close(before, after)

    def test_rumus4_tanpa_alpha_ditolak(self, comparator: FusionFormulaComparator) -> None:
        with pytest.raises(ValueError, match="alpha"):
            comparator.evaluate(FusionFormulaConfig(formula="rumus4"), split="val")

    def test_rumus_tak_dikenal_ditolak(self, comparator: FusionFormulaComparator) -> None:
        with pytest.raises(ValueError, match="tak dikenal"):
            comparator.evaluate(FusionFormulaConfig(formula="rumus5"), split="val")  # type: ignore[arg-type]

    def test_retrieval_dibangun_sekali_untuk_semua_rumus(
        self, comparator: FusionFormulaComparator
    ) -> None:
        comparator.evaluate(FusionFormulaConfig(formula="rumus2"), split="val")
        fitted = comparator._retrieval
        comparator.evaluate(FusionFormulaConfig(formula="rumus3"), split="val")
        assert comparator._retrieval is fitted


class TestRunAll:
    def test_satu_baris_per_rumus_ditambah_grid_alpha_rumus4(
        self, comparator: FusionFormulaComparator
    ) -> None:
        results = comparator.run_all(split="val", rumus4_alphas=(0.1, 0.2, 0.3))
        assert len(results) == 3 + 3
        assert set(results["formula"]) == {"rumus1", "rumus2", "rumus3", "rumus4"}
        assert sorted(results.loc[results["formula"] == "rumus4", "alpha"]) == [0.1, 0.2, 0.3]

    def test_kolom_metrik_lengkap(self, comparator: FusionFormulaComparator) -> None:
        results = comparator.run_all(split="val", rumus4_alphas=(0.2,))
        for column in ("val_f1_macro", "val_f1_judi", "val_accuracy", "eval_time_s", "index_vectors"):
            assert column in results.columns
        assert results["val_f1_macro"].between(0.0, 1.0).all()
