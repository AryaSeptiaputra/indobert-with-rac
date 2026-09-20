"""Test checksum isi tabel: portabel antar OS dan normalisasi ujung baris oleh git."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.checksum import (
    canonical_digest,
    canonical_rows,
    count_row_differences,
    load_text_table,
)
from src.utils.io import CorruptArtifactError


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "textOriginal": ["halo dunia", "baris satu\r\nbaris dua", "tanpa masalah"],
            "label": [0, 1, 0],
        }
    )


class TestCanonicalDigest:
    def test_hasil_deterministik(self, frame: pd.DataFrame) -> None:
        assert canonical_digest(frame) == canonical_digest(frame.copy())

    def test_crlf_di_dalam_sel_sama_dengan_lf(self, frame: pd.DataFrame) -> None:
        """Git dengan autocrlf membuang CR di dalam isi kolom teks saat commit."""
        as_committed = frame.assign(textOriginal=frame["textOriginal"].str.replace("\r\n", "\n"))
        assert canonical_digest(frame) == canonical_digest(as_committed)

    def test_cr_sendiri_sama_dengan_lf(self) -> None:
        with_cr = pd.DataFrame({"t": ["a\rb"]})
        with_lf = pd.DataFrame({"t": ["a\nb"]})
        assert canonical_digest(with_cr) == canonical_digest(with_lf)

    def test_label_angka_sama_dengan_label_teks(self, frame: pd.DataFrame) -> None:
        """Berkas CSV dibaca sebagai teks, sedangkan DataFrame baru berisi int."""
        assert canonical_digest(frame) == canonical_digest(frame.astype({"label": str}))

    def test_nilai_kosong_sama_dengan_string_kosong(self) -> None:
        assert canonical_digest(pd.DataFrame({"t": [np.nan]})) == canonical_digest(
            pd.DataFrame({"t": [""]})
        )

    def test_satu_karakter_berbeda_menghasilkan_hash_berbeda(self, frame: pd.DataFrame) -> None:
        changed = frame.copy()
        changed.loc[0, "textOriginal"] = "halo dunia!"
        assert canonical_digest(frame) != canonical_digest(changed)

    def test_urutan_baris_ikut_menentukan_hash(self, frame: pd.DataFrame) -> None:
        assert canonical_digest(frame) != canonical_digest(frame.iloc[::-1])

    def test_nama_kolom_ikut_menentukan_hash(self, frame: pd.DataFrame) -> None:
        assert canonical_digest(frame) != canonical_digest(frame.rename(columns={"label": "y"}))

    def test_karakter_non_ascii_dipertahankan(self) -> None:
        assert canonical_digest(pd.DataFrame({"t": ["ＤＯＲＡ７７"]})) != canonical_digest(
            pd.DataFrame({"t": ["DORA77"]})
        )

    def test_satu_baris_satu_string_json(self, frame: pd.DataFrame) -> None:
        rows = canonical_rows(frame)
        assert len(rows) == 3
        assert rows[1] == '["baris satu\\nbaris dua", "1"]'


class TestCountRowDifferences:
    def test_isi_sama_urutan_beda_memberi_nol(self, frame: pd.DataFrame) -> None:
        assert count_row_differences(frame, frame.iloc[::-1]) == (0, 0)

    def test_baris_yang_hanya_ada_di_satu_sisi_dihitung(self, frame: pd.DataFrame) -> None:
        old = frame.copy()
        old.loc[2, "textOriginal"] = "sudah diubah"
        assert count_row_differences(frame, old) == (1, 1)

    def test_baris_tambahan_hanya_ada_di_satu_sisi(self, frame: pd.DataFrame) -> None:
        assert count_row_differences(frame, frame.iloc[:2]) == (1, 0)


class TestLoadTextTable:
    def test_berkas_belum_ada_mengembalikan_none(self, tmp_path) -> None:
        assert load_text_table(tmp_path / "tidak_ada.csv") is None

    @pytest.mark.parametrize("terminator", ["\n", "\r\n"])
    def test_ujung_baris_berkas_tidak_memengaruhi_isi(self, tmp_path, frame, terminator) -> None:
        path = tmp_path / "data.csv"
        path.write_bytes(frame.to_csv(index=False, lineterminator=terminator).encode("utf-8"))
        assert canonical_digest(load_text_table(path)) == canonical_digest(frame)

    def test_berkas_hasil_normalisasi_git_tetap_identik(self, tmp_path, frame) -> None:
        """Salinan yang CR di dalam selnya sudah dibuang git tetap dianggap sama."""
        path = tmp_path / "data.csv"
        text = frame.to_csv(index=False, lineterminator="\r\n").replace("\r\n", "\n")
        path.write_bytes(text.encode("utf-8"))
        assert canonical_digest(load_text_table(path)) == canonical_digest(frame)

    def test_teks_yang_mirip_nan_tidak_diubah_menjadi_kosong(self, tmp_path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("t\nNA\nnull\n", encoding="utf-8")
        assert load_text_table(path)["t"].tolist() == ["NA", "null"]

    def test_csv_kosong_dilempar(self, tmp_path) -> None:
        path = tmp_path / "kosong.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(CorruptArtifactError):
            load_text_table(path)

    def test_bukan_utf8_dilempar(self, tmp_path) -> None:
        path = tmp_path / "latin.csv"
        path.write_bytes("t\nkafé\n".encode("latin-1"))
        with pytest.raises(CorruptArtifactError):
            load_text_table(path)
