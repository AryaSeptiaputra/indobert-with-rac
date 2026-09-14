"""Test I/O atomik dan pembedaan berkas hilang versus berkas rusak."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.utils.io import CorruptArtifactError, read_csv, read_json, write_csv, write_json


class TestReadJson:
    def test_berkas_belum_ada_mengembalikan_default(self, tmp_path) -> None:
        assert read_json(tmp_path / "belum_ada.json", default={"a": 1}) == {"a": 1}

    def test_berkas_belum_ada_tanpa_default_mengembalikan_none(self, tmp_path) -> None:
        assert read_json(tmp_path / "belum_ada.json") is None

    def test_membaca_json_yang_sah(self, tmp_path) -> None:
        path = tmp_path / "ok.json"
        path.write_text(json.dumps({"nilai": 42}), encoding="utf-8")
        assert read_json(path) == {"nilai": 42}

    def test_json_rusak_dilempar_bukan_diganti_default(self, tmp_path) -> None:
        """Menelan galat ini membuat pemanggil menimpa artefak yang masih bisa diselamatkan.

        Persis itulah yang dulu menghapus catatan model terbaik di best.json.
        """
        path = tmp_path / "rusak.json"
        path.write_text('{"belum ditutup": ', encoding="utf-8")
        with pytest.raises(CorruptArtifactError, match="JSON rusak"):
            read_json(path, default={})


class TestWriteJson:
    def test_membuat_folder_induk(self, tmp_path) -> None:
        path = write_json(tmp_path / "a" / "b" / "c.json", {"x": 1})
        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}

    def test_mempertahankan_karakter_non_ascii(self, tmp_path) -> None:
        path = write_json(tmp_path / "id.json", {"catatan": "coba naikkan lr"})
        assert "naikkan" in path.read_text(encoding="utf-8")

    def test_tidak_meninggalkan_berkas_sementara(self, tmp_path) -> None:
        write_json(tmp_path / "x.json", {"a": 1})
        assert list(tmp_path.glob("*.tmp")) == []

    def test_menimpa_isi_lama_sepenuhnya(self, tmp_path) -> None:
        path = tmp_path / "x.json"
        write_json(path, {"versi": 1, "usang": True})
        write_json(path, {"versi": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {"versi": 2}


class TestCsv:
    def test_berkas_belum_ada_mengembalikan_dataframe_kosong(self, tmp_path) -> None:
        assert read_csv(tmp_path / "belum_ada.csv").empty

    def test_pulang_pergi_mempertahankan_isi(self, tmp_path) -> None:
        frame = pd.DataFrame({"run_id": [1, 2], "val_f1_macro": [0.91, 0.93]})
        path = write_csv(tmp_path / "runs.csv", frame)
        pd.testing.assert_frame_equal(read_csv(path), frame)

    def test_tidak_menulis_kolom_index(self, tmp_path) -> None:
        path = write_csv(tmp_path / "x.csv", pd.DataFrame({"a": [1, 2]}))
        assert path.read_text(encoding="utf-8").splitlines()[0] == "a"

    def test_csv_kosong_mengembalikan_dataframe_kosong(self, tmp_path) -> None:
        path = tmp_path / "kosong.csv"
        path.write_text("", encoding="utf-8")
        assert read_csv(path).empty

    def test_csv_rusak_dilempar(self, tmp_path) -> None:
        path = tmp_path / "rusak.csv"
        path.write_text('a,b\n1,2\n3,4,5,6\n"tak ditutup', encoding="utf-8")
        with pytest.raises(CorruptArtifactError, match="CSV rusak"):
            read_csv(path)
