"""Checksum isi tabel yang tidak bergantung pada ujung baris dan sistem operasi.

SHA-256 atas BYTE berkas CSV tidak portabel. Pandas menulis ujung baris sesuai OS
(CRLF di Windows, LF di Linux), dan git dengan `core.autocrlf=true` menormalkan
CRLF menjadi LF saat commit, termasuk CR yang ada di DALAM isi kolom teks. Dua
berkas yang isinya sama bisa menghasilkan hash berbeda, dan gate reproduktibilitas
menolak split yang sebenarnya identik.

Modul ini menghitung hash atas ISI: setiap sel diubah ke teks, ujung baris di
dalam sel diseragamkan ke LF, lalu tiap baris diserialisasi sebagai JSON. Isi yang
berbeda (baris berbeda, urutan berbeda, nama kolom berbeda) tetap menghasilkan hash
berbeda.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.utils.io import CorruptArtifactError

# CRLF atau CR sendiri diubah ke LF. Contoh: "a\r\nb" dan "a\rb" menjadi "a\nb".
_NEWLINE_RE = re.compile(r"\r\n|\r")


def canonical_rows(frame: pd.DataFrame) -> list[str]:
    """Serialisasi kanonik: satu string JSON per baris.

    Args:
        frame: Tabel yang diserialisasi.

    Returns:
        Daftar string JSON, sesuai urutan baris. Nilai kosong (NaN) menjadi string
        kosong, angka menjadi teks, dan ujung baris di dalam sel menjadi LF.
    """
    text = frame.fillna("").astype(str)
    normalised = text.apply(lambda column: column.str.replace(_NEWLINE_RE, "\n", regex=True))
    return [
        json.dumps(list(row), ensure_ascii=False)
        for row in normalised.itertuples(index=False, name=None)
    ]


def canonical_digest(frame: pd.DataFrame) -> str:
    """SHA-256 atas isi kanonik tabel, termasuk nama kolom dan urutan baris.

    Args:
        frame: Tabel yang di-hash.

    Returns:
        Hash heksadesimal 64 karakter.
    """
    header = json.dumps(list(frame.columns), ensure_ascii=False)
    payload = "\n".join([header, *canonical_rows(frame)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def count_row_differences(new: pd.DataFrame, old: pd.DataFrame) -> tuple[int, int]:
    """Hitung baris yang hanya ada di satu sisi, tanpa memperhatikan urutan.

    Berguna untuk mendiagnosis hash yang berbeda: nol di kedua sisi berarti isinya
    sama dan hanya urutannya yang berubah.

    Args:
        new: Tabel baru.
        old: Tabel lama, dengan kolom yang sama.

    Returns:
        Tuple (baris hanya di `new`, baris hanya di `old`).
    """
    new_rows = Counter(canonical_rows(new))
    old_rows = Counter(canonical_rows(old))
    return sum((new_rows - old_rows).values()), sum((old_rows - new_rows).values())


def load_text_table(path: str | Path) -> pd.DataFrame | None:
    """Baca CSV apa adanya sebagai teks, tanpa konversi tipe atau NaN.

    Args:
        path: Lokasi berkas CSV.

    Returns:
        DataFrame berisi teks, atau `None` bila berkas belum ada.

    Raises:
        CorruptArtifactError: Kalau berkas ada tapi tidak bisa dibaca atau diurai.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
        raise CorruptArtifactError(f"CSV tidak bisa dibaca di {path}: {exc}") from exc
