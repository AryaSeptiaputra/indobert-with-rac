"""Test classification head dan mean pooling."""

from __future__ import annotations

import pytest
import torch

from src.models.heads import BERT_BASE_HIDDEN_SIZE, LinearHead, MLPHead, build_head, mean_pool


class TestBuildHead:
    def test_linear_menghasilkan_linear_head(self) -> None:
        assert isinstance(build_head("linear"), LinearHead)

    def test_mlp_menghasilkan_mlp_head(self) -> None:
        assert isinstance(build_head("mlp"), MLPHead)

    def test_arsitektur_tak_dikenal_ditolak(self) -> None:
        with pytest.raises(ValueError, match="arsitektur head tak dikenal"):
            build_head("transformer")

    @pytest.mark.parametrize("arch", ["linear", "mlp"])
    def test_bentuk_keluaran_batch_kali_kelas(self, arch: str) -> None:
        head = build_head(arch)
        output = head(torch.randn(7, BERT_BASE_HIDDEN_SIZE))
        assert output.shape == (7, 2)

    def test_jumlah_parameter_linear(self) -> None:
        head = build_head("linear", hidden_size=768)
        total = sum(p.numel() for p in head.parameters())
        assert total == 768 * 2 + 2

    def test_jumlah_parameter_mlp_sesuai_angka_bab_empat(self) -> None:
        """Konfigurasi RM-b terbaik: hidden_dim 1024 di atas masukan 768 dimensi."""
        head = build_head("mlp", hidden_size=768, hidden_dim=1024)
        total = sum(p.numel() for p in head.parameters())
        assert total == 768 * 1024 + 1024 + 1024 * 2 + 2
        assert total == 789_506

    def test_head_jauh_lebih_kecil_dari_encoder(self) -> None:
        """Inti klaim efisiensi RM-b terhadap 109 juta parameter RM-a."""
        head = build_head("mlp", hidden_size=768, hidden_dim=1024)
        total = sum(p.numel() for p in head.parameters())
        assert total / 109_485_314 < 0.01

    def test_dropout_diteruskan(self) -> None:
        head = build_head("linear", dropout=0.5)
        assert head.dropout.p == pytest.approx(0.5)

    def test_seluruh_parameter_head_trainable(self) -> None:
        head = build_head("mlp")
        assert all(p.requires_grad for p in head.parameters())

    def test_dropout_nonaktif_saat_eval(self) -> None:
        """Prediksi harus deterministik; dropout aktif membuatnya berubah-ubah."""
        head = build_head("linear", dropout=0.5).eval()
        features = torch.randn(4, BERT_BASE_HIDDEN_SIZE)
        torch.testing.assert_close(head(features), head(features))


class TestMeanPool:
    def test_bentuk_keluaran(self) -> None:
        pooled = mean_pool(torch.randn(3, 10, 8), torch.ones(3, 10, dtype=torch.long))
        assert pooled.shape == (3, 8)

    def test_token_padding_diabaikan(self) -> None:
        """Menyertakan padding akan mengencerkan representasi kalimat pendek."""
        hidden = torch.zeros(1, 4, 2)
        hidden[0, 0] = torch.tensor([1.0, 1.0])
        hidden[0, 1] = torch.tensor([3.0, 3.0])
        hidden[0, 2] = torch.tensor([100.0, 100.0])  # padding, harus diabaikan
        mask = torch.tensor([[1, 1, 0, 0]])
        torch.testing.assert_close(mean_pool(hidden, mask), torch.tensor([[2.0, 2.0]]))

    def test_mask_kosong_tidak_membagi_nol(self) -> None:
        pooled = mean_pool(torch.randn(1, 4, 3), torch.zeros(1, 4, dtype=torch.long))
        assert torch.isfinite(pooled).all()

    def test_setara_rata_rata_biasa_saat_tanpa_padding(self) -> None:
        hidden = torch.randn(2, 5, 4)
        mask = torch.ones(2, 5, dtype=torch.long)
        torch.testing.assert_close(mean_pool(hidden, mask), hidden.mean(dim=1))
