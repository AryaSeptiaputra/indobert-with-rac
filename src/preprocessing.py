"""Preprocessing untuk deteksi komentar promosi judi (IndoBERT-with-RAC).

Mengeksekusi keputusan fase EDA (lihat EDA_REPORT_BAGIAN3.md §6). Modul ini
murni pandas/sklearn (tanpa torch) agar bisa dipakai ulang lintas notebook.

Alur yang didukung:
    load -> drop missing -> resolve konflik label -> dedup (NFKC-exact)
    -> stratified split 70/15/15 -> clean_text (transformasi setelah split)

Catatan penting:
- NFKC dipakai sebagai fondasi normalisasi & kunci dedup (menangkap obfuskasi
  Unicode kelas 1 seperti fullwidth/double-struck/enclosed).
- Placeholder [URL]/[MENTION]/[NUM] didaftarkan sebagai special token di
  tokenizer (lihat src/dataset.py); di sini hanya penulisan string-nya.
- Angka yang diganti hanya token digit *berdiri sendiri* (\\b\\d+\\b) — brand
  alfanumerik seperti DORA77 / PROBET855 sengaja dipertahankan utuh.
- Tidak ada lowercase manual (tokenizer p2 do_lower_case=True), tidak ada
  penghapusan emoji/punktuasi.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

RANDOM_SEED = 42

# Placeholder didaftarkan sebagai additional_special_tokens di tokenizer.
URL_PLACEHOLDER = "[URL]"
MENTION_PLACEHOLDER = "[MENTION]"
NUM_PLACEHOLDER = "[NUM]"
SPECIAL_TOKENS = [URL_PLACEHOLDER, MENTION_PLACEHOLDER, NUM_PLACEHOLDER]

# --- Normalisasi Unicode (fondasi) ---------------------------------------

# Karakter tak-terlihat dibuang sebelum NFKC. Sumber dari daftar codepoint agar
# file ini 100% ASCII (tahan korupsi editor/git). Identik dengan sel 2.1 EDA.
_INVIS_CP = [
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F,  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    0x2060, 0x00AD,                          # word joiner, soft hyphen
    *range(0xFE00, 0xFE10),                  # variation selectors VS1-VS16
]
_INVISIBLE = re.compile("[" + "".join(chr(c) for c in _INVIS_CP) + "]")


def normalize_nfkc(text) -> str:
    """Buang karakter tak-terlihat lalu terapkan NFKC.

    Memetakan varian Unicode (math bold/italic, fullwidth, double-struck,
    enclosed) ke ASCII bila ada decomposition. Small caps tidak punya
    decomposition -> bertahan (residual, didokumentasikan sebagai future work).
    """
    return unicodedata.normalize("NFKC", _INVISIBLE.sub("", str(text)))


# --- Transformasi teks (placeholder) -------------------------------------

_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|t\.me/\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_NUM_RE = re.compile(r"\b\d+\b")          # hanya angka berdiri sendiri
_WS_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """NFKC -> placeholder URL/mention/angka -> rapikan whitespace.

    Tidak melakukan lowercase (tokenizer uncased), tidak menghapus emoji atau
    tanda baca. Brand alfanumerik (DORA77) dipertahankan karena _NUM_RE hanya
    menyasar token digit murni.
    """
    t = normalize_nfkc(text)
    t = _URL_RE.sub(URL_PLACEHOLDER, t)
    t = _MENTION_RE.sub(MENTION_PLACEHOLDER, t)
    t = _NUM_RE.sub(NUM_PLACEHOLDER, t)
    return _WS_RE.sub(" ", t).strip()


# --- Deduplikasi & konflik label -----------------------------------------

def resolve_label_conflicts(df: pd.DataFrame, key: str = "nfkc_key",
                            label: str = "label") -> pd.DataFrame:
    """Grup `key` dengan >1 label unik -> set label 1 (default riset).

    Dijalankan sebelum dedup agar baris yang dipertahankan berlabel 1.
    Mengembalikan salinan df; tidak memodifikasi input.
    """
    df = df.copy()
    grp = df.groupby(key)[label].transform("nunique")
    df.loc[grp > 1, label] = 1
    return df


def deduplicate(df: pd.DataFrame, text_col: str = "textOriginal",
                label: str = "label") -> pd.DataFrame:
    """Dedup dengan kunci NFKC-exact (keep first), setelah resolusi konflik label.

    Menambah kolom `nfkc_key`. Mengembalikan df ter-dedup (index di-reset).
    """
    df = df.copy()
    df["nfkc_key"] = df[text_col].map(normalize_nfkc)
    df = resolve_label_conflicts(df, key="nfkc_key", label=label)
    df = df.drop_duplicates(subset="nfkc_key", keep="first").reset_index(drop=True)
    return df


# --- Split & class weight -------------------------------------------------

def stratified_split(df: pd.DataFrame, label: str = "label",
                     seed: int = RANDOM_SEED,
                     ratios=(0.70, 0.15, 0.15)):
    """Stratified split 70/15/15 (default). Mengembalikan (train, val, test).

    Dua tahap: train vs sisa, lalu sisa dibagi val/test proporsional.
    """
    train_r, val_r, test_r = ratios
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios harus berjumlah 1.0"

    train_df, rest_df = train_test_split(
        df, train_size=train_r, stratify=df[label], random_state=seed,
    )
    # proporsi test di dalam sisa
    test_within_rest = test_r / (val_r + test_r)
    val_df, test_df = train_test_split(
        rest_df, test_size=test_within_rest, stratify=rest_df[label],
        random_state=seed,
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def compute_class_weights(y, classes=(0, 1)) -> dict:
    """Class weight 'balanced' (sklearn) dihitung dari label train saja.

    Mengembalikan dict {kelas: bobot} agar mudah diserialkan ke JSON dan
    dipakai pada loss (mis. CrossEntropyLoss(weight=...)).
    """
    classes = np.array(classes)
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(y))
    return {int(c): float(w) for c, w in zip(classes, weights)}
