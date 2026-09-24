"""Format angka untuk gambar: koma desimal, titik ribuan, pembulatan setengah ke atas.

Pembulatan bawaan Python (dan f-string) memakai representasi biner float,
sehingga 0,125 bisa menjadi 0,12. Anotasi gambar memakai `Decimal` dengan
`ROUND_HALF_UP` supaya angka di gambar sama dengan pembulatan manual di tabel.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def round_half_up(value: float, decimals: int) -> Decimal:
    """Bulatkan setengah ke atas berdasarkan representasi desimal terpendek float."""
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def format_number(value: float, decimals: int = 2, sign: bool = False) -> str:
    """Format gaya Indonesia: `1.234,57`; `sign=True` memberi `+` pada nilai positif.

    Nilai yang dibulatkan menjadi nol ditulis tanpa tanda minus.
    """
    rounded = round_half_up(value, decimals)
    if rounded == 0:
        rounded = abs(rounded)
    text = f"{rounded:,.{decimals}f}"
    if sign and rounded > 0:
        text = "+" + text
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


__all__ = ["format_number", "round_half_up"]
