"""Test arsip hasil kampanye: isi, pengecualian folder besar, dan penamaan."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from src.services.export import NO_HARDWARE_LABEL, ResultArchiver


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    """Folder hasil tiruan: berkas kecil yang harus diamankan dan folder besar yang tidak."""
    root = tmp_path / "outputs"
    (root / "tuning" / "history").mkdir(parents=True)
    (root / "tuning" / "checkpoints" / "rmb_heads").mkdir(parents=True)
    (root / "tuning" / "features" / "encoder").mkdir(parents=True)
    (root / "baseline").mkdir()

    # write_bytes: write_text di Windows mengubah LF menjadi CRLF dan menggeser isi yang diuji.
    (root / "tuning" / "runs_rmb.csv").write_bytes(b"run_id\n1\n")
    (root / "tuning" / "best.json").write_text("{}", encoding="utf-8")
    (root / "tuning" / "history" / "rmb_history.csv").write_text("epoch\n1\n", encoding="utf-8")
    (root / "tuning" / "hardware.json").write_text(
        json.dumps({"gpu": "NVIDIA GeForce RTX 3090"}), encoding="utf-8"
    )
    (root / "tuning" / "checkpoints" / "rma_best.pt").write_bytes(b"x" * 1000)
    (root / "tuning" / "checkpoints" / "rmb_heads" / "run_1.pt").write_bytes(b"y" * 100)
    (root / "tuning" / "features" / "encoder" / "train_emb.npy").write_bytes(b"z" * 500)
    (root / "baseline" / "runs_rma.csv").write_text("run_id\n1\n", encoding="utf-8")
    return root


def names_in(archive: Path) -> set[str]:
    with tarfile.open(archive) as handle:
        return set(handle.getnames())


class TestResultArchiver:
    def test_berkas_hasil_kecil_ikut_diarsipkan(self, outputs: Path, tmp_path: Path) -> None:
        result = ResultArchiver(outputs, tmp_path / "out").archive()

        assert {
            "outputs/tuning/runs_rmb.csv",
            "outputs/tuning/best.json",
            "outputs/tuning/history/rmb_history.csv",
            "outputs/tuning/hardware.json",
            "outputs/baseline/runs_rma.csv",
        } <= names_in(result.path)

    def test_checkpoint_dan_fitur_tidak_ikut_secara_bawaan(self, outputs: Path, tmp_path: Path) -> None:
        names = names_in(ResultArchiver(outputs, tmp_path / "out").archive().path)

        assert not any("checkpoints" in name or "features" in name for name in names)

    def test_checkpoint_dan_fitur_ikut_bila_diminta(self, outputs: Path, tmp_path: Path) -> None:
        result = ResultArchiver(outputs, tmp_path / "out", include_heavy=True).archive()

        assert "outputs/tuning/checkpoints/rma_best.pt" in names_in(result.path)
        assert "outputs/tuning/checkpoints/rmb_heads/run_1.pt" in names_in(result.path)
        assert "outputs/tuning/features/encoder/train_emb.npy" in names_in(result.path)

    def test_berkas_tambahan_ada_di_akar_arsip(self, outputs: Path, tmp_path: Path) -> None:
        excel = tmp_path / "HASIL.xlsx"
        excel.write_bytes(b"xlsx")

        names = names_in(ResultArchiver(outputs, tmp_path / "out").archive([excel]).path)

        assert "HASIL.xlsx" in names

    def test_berkas_tambahan_yang_belum_ada_dilewati(self, outputs: Path, tmp_path: Path) -> None:
        result = ResultArchiver(outputs, tmp_path / "out").archive([tmp_path / "HASIL.xlsx"])
        assert "HASIL.xlsx" not in names_in(result.path)

    def test_nama_memuat_gpu_dan_waktu(self, outputs: Path, tmp_path: Path) -> None:
        name = ResultArchiver(outputs, tmp_path / "out").archive().path.name
        assert name.startswith("hasil_nvidia-geforce-rtx-3090_")
        assert name.endswith(".tar.gz")

    def test_tanpa_hardware_json_memakai_label_bawaan(self, outputs: Path, tmp_path: Path) -> None:
        (outputs / "tuning" / "hardware.json").unlink()
        name = ResultArchiver(outputs, tmp_path / "out").archive().path.name
        assert name.startswith(f"hasil_{NO_HARDWARE_LABEL}_")

    def test_hasil_ringkasan_sesuai_isi_arsip(self, outputs: Path, tmp_path: Path) -> None:
        result = ResultArchiver(outputs, tmp_path / "out").archive()

        assert result.n_files == len(names_in(result.path))
        assert result.size_bytes == result.path.stat().st_size

    def test_isi_berkas_terbaca_kembali_utuh(self, outputs: Path, tmp_path: Path) -> None:
        result = ResultArchiver(outputs, tmp_path / "out").archive()

        with tarfile.open(result.path) as handle:
            content = handle.extractfile("outputs/tuning/runs_rmb.csv").read().decode("utf-8")
        assert content == "run_id\n1\n"

    def test_tidak_meninggalkan_berkas_sementara(self, outputs: Path, tmp_path: Path) -> None:
        ResultArchiver(outputs, tmp_path / "out").archive()
        assert not list((tmp_path / "out").glob("*.tmp"))

    def test_folder_hasil_tidak_ada_ditolak(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="tidak ditemukan"):
            ResultArchiver(tmp_path / "tidak_ada", tmp_path / "out").archive()

    def test_folder_hasil_kosong_ditolak(self, tmp_path: Path) -> None:
        (tmp_path / "outputs").mkdir()
        with pytest.raises(FileNotFoundError, match="tidak ada hasil"):
            ResultArchiver(tmp_path / "outputs", tmp_path / "out").archive()

    def test_hanya_folder_besar_dianggap_kosong(self, outputs: Path, tmp_path: Path) -> None:
        empty = tmp_path / "hanya_besar"
        (empty / "tuning" / "checkpoints").mkdir(parents=True)
        (empty / "tuning" / "checkpoints" / "a.pt").write_bytes(b"x")
        with pytest.raises(FileNotFoundError, match="tidak ada hasil"):
            ResultArchiver(empty, tmp_path / "out").archive()

    def test_tujuan_di_dalam_folder_hasil_ditolak(self, outputs: Path) -> None:
        """Arsip yang ditulis ke dalam folder yang sedang diarsipkan akan memuat dirinya sendiri."""
        with pytest.raises(ValueError, match="tidak boleh berada di dalam"):
            ResultArchiver(outputs, outputs / "arsip")

    def test_folder_tujuan_dibuat_bila_belum_ada(self, outputs: Path, tmp_path: Path) -> None:
        destination = tmp_path / "dalam" / "sekali"
        ResultArchiver(outputs, destination).archive()
        assert destination.is_dir()
