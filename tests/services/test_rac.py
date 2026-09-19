"""Test RACClassifier: fusi probabilitas, distribusi retrieval, dan guard indeks."""

from __future__ import annotations

import numpy as np
import pytest

from src.services.rac import NeighborCache, RACClassifier, l2_normalize, softmax


class TestUtilitasNumerik:
    def test_l2_normalize_menghasilkan_norma_satu(self, rng) -> None:
        matrix = rng.normal(size=(20, 8))
        normalized = l2_normalize(matrix)
        np.testing.assert_allclose(
            np.linalg.norm(normalized, axis=1), 1.0, rtol=1e-6
        )

    def test_l2_normalize_menghasilkan_float32_kontigu(self, rng) -> None:
        normalized = l2_normalize(rng.normal(size=(5, 4)))
        assert normalized.dtype == np.float32
        assert normalized.flags["C_CONTIGUOUS"]

    def test_l2_normalize_tidak_membagi_nol(self) -> None:
        normalized = l2_normalize(np.zeros((2, 3), dtype=np.float32))
        assert np.isfinite(normalized).all()

    def test_softmax_berjumlah_satu(self, rng) -> None:
        probabilities = softmax(rng.normal(size=(15, 2)) * 10)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)

    def test_softmax_stabil_untuk_logit_besar(self) -> None:
        probabilities = softmax(np.array([[1000.0, 999.0]]))
        assert np.isfinite(probabilities).all()


class TestRACClassifier:
    def test_alpha_di_luar_rentang_ditolak(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            RACClassifier(alpha=1.5)

    def test_k_nol_ditolak(self) -> None:
        with pytest.raises(ValueError, match="k harus"):
            RACClassifier(k=0)

    def test_weighting_tak_dikenal_ditolak(self) -> None:
        with pytest.raises(ValueError, match="weighting"):
            RACClassifier(weighting="cosine")

    def test_predict_sebelum_fit_ditolak(self, rng) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            RACClassifier().predict(rng.normal(size=(3, 4)), softmax(rng.normal(size=(3, 2))))

    def test_k_melebihi_ukuran_indeks_ditolak(self, rng) -> None:
        """FAISS membalas kekurangan tetangga dengan indeks -1.

        Tanpa guard ini, pengindeksan numpy menerjemahkan -1 menjadi label train
        TERAKHIR: tetangga yang salah, tanpa satu pun pesan galat.
        """
        embeddings = rng.normal(size=(5, 4)).astype(np.float32)
        labels = rng.integers(0, 2, 5)
        classifier = RACClassifier(k=6).fit(embeddings, labels)
        with pytest.raises(ValueError, match="melebihi jumlah vektor"):
            classifier.retrieve(embeddings)

    def test_jumlah_label_harus_cocok(self, rng) -> None:
        with pytest.raises(ValueError, match="jumlah embedding"):
            RACClassifier().fit(rng.normal(size=(5, 4)), np.zeros(3))

    def test_embedding_kosong_ditolak(self) -> None:
        with pytest.raises(ValueError, match="kosong"):
            RACClassifier().fit(np.zeros((0, 4), dtype=np.float32), np.zeros(0))

    def test_index_size_sama_dengan_jumlah_train(self, rng) -> None:
        classifier = RACClassifier().fit(
            rng.normal(size=(50, 8)).astype(np.float32), rng.integers(0, 2, 50)
        )
        assert classifier.index_size == 50

    def test_alpha_nol_setara_rm_b_murni(self, rng) -> None:
        """alpha=0 harus membuang cabang retrieval sepenuhnya."""
        train = rng.normal(size=(40, 8)).astype(np.float32)
        labels = rng.integers(0, 2, 40)
        query = rng.normal(size=(10, 8)).astype(np.float32)
        p_bert = softmax(rng.normal(size=(10, 2)))

        classifier = RACClassifier(alpha=0.0, k=5).fit(train, labels)
        predictions, p_final = classifier.predict(query, p_bert)

        np.testing.assert_allclose(p_final, p_bert, rtol=1e-6)
        np.testing.assert_array_equal(predictions, p_bert.argmax(axis=1))

    def test_alpha_satu_membuang_cabang_bert(self, rng) -> None:
        train = rng.normal(size=(40, 8)).astype(np.float32)
        labels = rng.integers(0, 2, 40)
        query = rng.normal(size=(10, 8)).astype(np.float32)

        classifier = RACClassifier(alpha=1.0, k=5).fit(train, labels)
        _, with_bert = classifier.predict(query, softmax(rng.normal(size=(10, 2))))
        _, without_bert = classifier.predict(query, np.zeros((10, 2), dtype=np.float32))

        np.testing.assert_allclose(with_bert, without_bert, rtol=1e-6)

    def test_p_final_tetap_distribusi_sah(self, rng) -> None:
        """Kedua masukan adalah distribusi dan bobot fusi berjumlah satu.

        Karena itu hasil fusi sudah menjadi distribusi sah, dan softmax kedua
        tidak boleh diterapkan setelahnya.
        """
        train = rng.normal(size=(40, 8)).astype(np.float32)
        classifier = RACClassifier(alpha=0.3, k=5).fit(train, rng.integers(0, 2, 40))
        _, p_final = classifier.predict(
            rng.normal(size=(10, 8)).astype(np.float32), softmax(rng.normal(size=(10, 2)))
        )
        np.testing.assert_allclose(p_final.sum(axis=1), 1.0, rtol=1e-6)
        assert (p_final >= 0).all()

    def test_distribusi_retrieval_berjumlah_satu(self) -> None:
        classifier = RACClassifier(k=3)
        similarities = np.array([[0.9, 0.5, 0.1], [0.2, 0.2, 0.2]], dtype=np.float32)
        neighbours = np.array([[1, 1, 0], [0, 0, 1]])
        distribution = classifier.retrieval_distribution(similarities, neighbours)
        np.testing.assert_allclose(distribution.sum(axis=1), 1.0, rtol=1e-6)

    def test_pembobotan_similarity_mengutamakan_tetangga_terdekat(self) -> None:
        classifier = RACClassifier(k=2, weighting="similarity")
        similarities = np.array([[0.9, 0.1]], dtype=np.float32)
        neighbours = np.array([[1, 0]])
        distribution = classifier.retrieval_distribution(similarities, neighbours)
        assert distribution[0, 1] == pytest.approx(0.9)

    def test_pembobotan_uniform_mengabaikan_jarak(self) -> None:
        classifier = RACClassifier(k=2, weighting="uniform")
        similarities = np.array([[0.9, 0.1]], dtype=np.float32)
        neighbours = np.array([[1, 0]])
        distribution = classifier.retrieval_distribution(similarities, neighbours)
        np.testing.assert_allclose(distribution[0], [0.5, 0.5], rtol=1e-6)

    def test_semua_similarity_nol_jatuh_ke_seragam(self) -> None:
        """Tanpa fallback ini, baris tersebut akan menghasilkan pembagian nol."""
        classifier = RACClassifier(k=2, weighting="similarity")
        similarities = np.array([[-0.5, -0.9]], dtype=np.float32)
        neighbours = np.array([[1, 0]])
        distribution = classifier.retrieval_distribution(similarities, neighbours)
        np.testing.assert_allclose(distribution[0], [0.5, 0.5], rtol=1e-6)

    def test_retrieval_menemukan_dirinya_sendiri(self, rng) -> None:
        """Query yang identik dengan satu vektor train harus jadi tetangga terdekat.

        Inilah alasan indeks hanya boleh dibangun dari split train: kalau
        near-duplicate train/test lolos, retrieval menemukan jawabannya sendiri.
        """
        train = rng.normal(size=(30, 8)).astype(np.float32)
        labels = np.arange(30) % 2
        classifier = RACClassifier(k=1).fit(train, labels)
        _, indices = classifier.retrieve(train[:5])
        np.testing.assert_array_equal(indices[:, 0], np.arange(5))

    def test_bentuk_p_bert_tak_cocok_ditolak(self, rng) -> None:
        classifier = RACClassifier(k=3).fit(
            rng.normal(size=(20, 8)).astype(np.float32), rng.integers(0, 2, 20)
        )
        with pytest.raises(ValueError, match="jumlah baris p_bert"):
            classifier.predict(
                rng.normal(size=(10, 8)).astype(np.float32),
                softmax(rng.normal(size=(5, 2))),
            )


class TestNeighborCache:
    @pytest.fixture
    def corpus(self, rng):
        return rng.normal(size=(80, 8)).astype(np.float32), rng.integers(0, 2, 80)

    def test_potongan_sama_dengan_pencarian_langsung_dengan_k_itu(self, corpus, rng) -> None:
        embeddings, labels = corpus
        queries = rng.normal(size=(12, 8)).astype(np.float32)
        cache = NeighborCache(embeddings, labels, max_k=20)

        similarities, neighbor_labels = cache.neighbors("val", queries, k=5)
        direct = RACClassifier(k=5).fit(embeddings, labels)
        direct_sims, direct_idx = direct.retrieve(queries)

        np.testing.assert_allclose(similarities, direct_sims, rtol=1e-6)
        np.testing.assert_array_equal(neighbor_labels, labels[direct_idx])

    def test_pencarian_hanya_sekali_per_split(self, corpus, rng) -> None:
        embeddings, labels = corpus
        cache = NeighborCache(embeddings, labels, max_k=10)
        first = cache.neighbors("val", rng.normal(size=(6, 8)).astype(np.float32), k=3)

        # Embedding berbeda dengan nama split sama tidak dihitung ulang: kunci cache
        # adalah nama split, dan itu memang kontraknya.
        second = cache.neighbors("val", rng.normal(size=(6, 8)).astype(np.float32), k=3)
        np.testing.assert_array_equal(first[1], second[1])

    def test_k_di_luar_rentang_ditolak(self, corpus, rng) -> None:
        embeddings, labels = corpus
        cache = NeighborCache(embeddings, labels, max_k=5)
        queries = rng.normal(size=(3, 8)).astype(np.float32)
        with pytest.raises(ValueError, match="di luar rentang"):
            cache.neighbors("val", queries, k=6)
        with pytest.raises(ValueError, match="di luar rentang"):
            cache.neighbors("val", queries, k=0)

    def test_ukuran_indeks_sama_dengan_jumlah_train(self, corpus) -> None:
        embeddings, labels = corpus
        assert NeighborCache(embeddings, labels, max_k=3).index_size == 80
