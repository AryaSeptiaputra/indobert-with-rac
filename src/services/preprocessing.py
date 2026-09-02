"""Pembersihan teks dan pembangunan split siap latih.

Mengeksekusi keputusan fase EDA (lihat EDA_REPORT_BAGIAN3.md bagian 6):

    muat -> buang missing -> resolusi konflik label -> dedup (NFKC-exact)
    -> stratified split 70/15/15 -> clean_text (dijalankan SETELAH split)

NFKC dipakai sebagai fondasi normalisasi sekaligus kunci dedup karena menangkap
obfuskasi Unicode (fullwidth, double-struck, enclosed). Angka yang diganti hanya
token digit berdiri sendiri, sehingga brand alfanumerik seperti DORA77 tetap
utuh. Tidak ada lowercase manual (tokenizer p2 sudah uncased) dan tidak ada
penghapusan emoji maupun tanda baca.

Urutan operasi di modul ini menentukan isi split, jadi mengubahnya membatalkan
seluruh angka eksperimen yang sudah dihasilkan.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    LABEL_COLUMN,
    MENTION_PLACEHOLDER,
    NUM_PLACEHOLDER,
    PROJECT_ROOT,
    RAW_TEXT_COLUMN,
    SPLIT_NAMES,
    TEXT_COLUMN,
    URL_PLACEHOLDER,
    settings,
)
from src.utils.io import write_csv, write_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEDUP_KEY_COLUMN = "nfkc_key"

# Codepoint karakter tak-terlihat yang dibuang sebelum NFKC. Ditulis sebagai
# daftar angka agar berkas ini tetap 100% ASCII dan tahan korupsi editor.
_INVISIBLE_CODEPOINTS: tuple[int, ...] = (
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F,  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    0x2060, 0x00AD,                          # word joiner, soft hyphen
    *range(0xFE00, 0xFE10),                  # variation selector VS1-VS16
)


class TextCleaner:
    """Normalisasi Unicode dan penggantian placeholder untuk komentar YouTube.

    Regex dikompilasi sekali di konstruktor lalu dipakai ulang untuk seluruh
    9.395 baris, sehingga pembersihan satu split tidak mengompilasi ulang pola
    yang sama ribuan kali.
    """

    def __init__(
        self,
        url_placeholder: str = URL_PLACEHOLDER,
        mention_placeholder: str = MENTION_PLACEHOLDER,
        num_placeholder: str = NUM_PLACEHOLDER,
    ) -> None:
        self.url_placeholder = url_placeholder
        self.mention_placeholder = mention_placeholder
        self.num_placeholder = num_placeholder

        invisible_class = "".join(chr(cp) for cp in _INVISIBLE_CODEPOINTS)
        self._invisible_re = re.compile(f"[{invisible_class}]")
        # URL http/https, bare www, dan tautan Telegram (kanal promosi judi).
        self._url_re = re.compile(r"(?:https?://\S+|www\.\S+|t\.me/\S+)", re.IGNORECASE)
        self._mention_re = re.compile(r"@\w+")
        # Hanya angka berdiri sendiri; DORA77 dan PROBET855 sengaja tidak kena.
        self._num_re = re.compile(r"\b\d+\b")
        self._whitespace_re = re.compile(r"\s+")

    def normalize_nfkc(self, text: object) -> str:
        """Buang karakter tak-terlihat lalu terapkan normalisasi NFKC.

        Varian Unicode (math bold/italic, fullwidth, double-struck, enclosed)
        dipetakan ke ASCII bila punya decomposition. Small caps tidak punya
        decomposition sehingga bertahan; ini residual yang diketahui.

        Args:
            text: Nilai apa pun; dikonversi ke `str` lebih dulu.

        Returns:
            Teks hasil normalisasi.
        """
        return unicodedata.normalize("NFKC", self._invisible_re.sub("", str(text)))

    def clean(self, text: object) -> str:
        """NFKC, ganti URL/mention/angka dengan placeholder, rapikan whitespace.

        Args:
            text: Teks komentar mentah.

        Returns:
            Teks bersih siap ditokenisasi.
        """
        cleaned = self.normalize_nfkc(text)
        cleaned = self._url_re.sub(self.url_placeholder, cleaned)
        cleaned = self._mention_re.sub(self.mention_placeholder, cleaned)
        cleaned = self._num_re.sub(self.num_placeholder, cleaned)
        return self._whitespace_re.sub(" ", cleaned).strip()

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Terapkan `clean` ke seluruh elemen Series.

        Args:
            series: Kolom teks mentah.

        Returns:
            Series berisi teks bersih, index dipertahankan.
        """
        return series.map(self.clean)


class DatasetBuilder:
    """Bangun split train/val/test siap latih dari CSV berlabel mentah.

    Args:
        cleaner: Pembersih teks; `None` membuat `TextCleaner` default.
        seed: Seed split; `None` memakai `settings.random_seed`.
        ratios: Proporsi (train, val, test); harus berjumlah 1.

    Raises:
        ValueError: Kalau `ratios` tidak berjumlah 1.
    """

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        seed: int | None = None,
        ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    ) -> None:
        if abs(sum(ratios) - 1.0) >= 1e-9:
            raise ValueError(f"ratios harus berjumlah 1,0; diterima {ratios} = {sum(ratios)}")

        self.cleaner = cleaner or TextCleaner()
        self.seed = settings.random_seed if seed is None else seed
        self.ratios = ratios
        self.counts: dict[str, int] = {}
        self.label_conflict: dict[str, object] = {}

    def resolve_label_conflicts(
        self,
        frame: pd.DataFrame,
        key: str = DEDUP_KEY_COLUMN,
        label: str = LABEL_COLUMN,
    ) -> pd.DataFrame:
        """Setel label 1 untuk seluruh grup `key` yang punya lebih dari satu label.

        Dijalankan sebelum dedup agar baris yang bertahan berlabel 1. Kebijakan
        ini konservatif untuk deteksi: teks yang pernah dilabeli judi tetap
        dianggap judi.

        Args:
            frame: DataFrame yang sudah punya kolom `key`.
            key: Nama kolom kunci dedup.
            label: Nama kolom label.

        Returns:
            Salinan DataFrame; masukan tidak dimodifikasi.

        Raises:
            KeyError: Kalau kolom `key` atau `label` tidak ada.
        """
        for column in (key, label):
            if column not in frame.columns:
                raise KeyError(f"kolom {column!r} tidak ada di DataFrame")

        frame = frame.copy()
        unique_labels = frame.groupby(key)[label].transform("nunique")
        conflicted = unique_labels > 1
        self.label_conflict = {
            "groups": int(frame.loc[conflicted, key].nunique()),
            "rows": int(conflicted.sum()),
            "policy": "assign_1",
        }
        frame.loc[conflicted, label] = 1
        return frame

    def deduplicate(
        self,
        frame: pd.DataFrame,
        text_col: str = RAW_TEXT_COLUMN,
        label: str = LABEL_COLUMN,
    ) -> pd.DataFrame:
        """Dedup dengan kunci NFKC-exact (keep first), setelah resolusi konflik label.

        Args:
            frame: DataFrame mentah.
            text_col: Nama kolom teks mentah.
            label: Nama kolom label.

        Returns:
            DataFrame ter-dedup dengan kolom `nfkc_key`, index di-reset.

        Raises:
            KeyError: Kalau kolom teks tidak ada.
        """
        if text_col not in frame.columns:
            raise KeyError(f"kolom teks {text_col!r} tidak ada di DataFrame")

        frame = frame.copy()
        frame[DEDUP_KEY_COLUMN] = frame[text_col].map(self.cleaner.normalize_nfkc)
        frame = self.resolve_label_conflicts(frame, key=DEDUP_KEY_COLUMN, label=label)
        deduped = frame.drop_duplicates(subset=DEDUP_KEY_COLUMN, keep="first").reset_index(
            drop=True
        )
        logger.info("Dedup NFKC-exact: %d -> %d baris", len(frame), len(deduped))
        return deduped

    def stratified_split(
        self,
        frame: pd.DataFrame,
        label: str = LABEL_COLUMN,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split stratifikasi dua tahap: train vs sisa, lalu sisa jadi val/test.

        Args:
            frame: DataFrame yang sudah ter-dedup.
            label: Nama kolom label untuk stratifikasi.

        Returns:
            Tuple (train, val, test) dengan index masing-masing di-reset.

        Raises:
            ValueError: Kalau ada kelas dengan anggota terlalu sedikit untuk
                distratifikasi (dilempar oleh scikit-learn).
        """
        train_ratio, val_ratio, test_ratio = self.ratios

        train_df, rest_df = train_test_split(
            frame,
            train_size=train_ratio,
            stratify=frame[label],
            random_state=self.seed,
        )
        test_within_rest = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            rest_df,
            test_size=test_within_rest,
            stratify=rest_df[label],
            random_state=self.seed,
        )

        splits = (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
        logger.info(
            "Stratified split %s: train=%d val=%d test=%d",
            self.ratios,
            len(splits[0]),
            len(splits[1]),
            len(splits[2]),
        )
        return splits

    def compute_class_weights(
        self,
        labels: pd.Series | np.ndarray,
        classes: tuple[int, ...] = (0, 1),
    ) -> dict[int, float]:
        """Hitung class weight 'balanced' dari label TRAIN saja.

        Menghitungnya dari split lain akan membocorkan distribusi val/test ke
        dalam loss.

        Args:
            labels: Label split train.
            classes: Kelas yang diperhitungkan.

        Returns:
            Dict {kelas: bobot}, siap diserialkan ke JSON dan dipakai sebagai
            `weight` pada `CrossEntropyLoss`.

        Raises:
            ValueError: Kalau ada kelas di `classes` yang tidak muncul di `labels`.
        """
        class_array = np.array(classes)
        weights = compute_class_weight(
            "balanced", classes=class_array, y=np.asarray(labels)
        )
        return {int(c): float(w) for c, w in zip(class_array, weights)}

    def find_leakage(
        self,
        splits: dict[str, pd.DataFrame],
        key: str = DEDUP_KEY_COLUMN,
    ) -> dict[str, int]:
        """Hitung irisan kunci dedup antar split; seluruhnya harus nol.

        Kebocoran train-test membuat retrieval RM-c "curang" karena tetangga
        terdekat sebuah sampel test bisa jadi dirinya sendiri di indeks train.

        Args:
            splits: Peta nama split ke DataFrame-nya.
            key: Nama kolom kunci dedup.

        Returns:
            Dict {"train-val": n, "train-test": n, "val-test": n}.
        """
        names = list(splits)
        overlaps: dict[str, int] = {}
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                shared = set(splits[left][key]) & set(splits[right][key])
                overlaps[f"{left}-{right}"] = len(shared)
        return overlaps

    def drop_cross_split_duplicates(
        self,
        splits: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], int]:
        """Buang baris yang `text_clean`-nya sudah muncul di split berprioritas lebih tinggi.

        Dedup NFKC di awal bekerja pada teks ASLI, sehingga dua komentar yang
        hanya berbeda pada URL atau angka masih lolos sebagai baris terpisah.
        Setelah `clean_text` mengganti keduanya dengan placeholder, teksnya jadi
        identik dan menjadi kebocoran nyata: sampel test yang teks efektifnya
        sudah dilihat model saat latih.

        Prioritas train > val > test; train tidak pernah dikurangi, sehingga
        himpunan latih tetap utuh dan yang menyusut hanya himpunan evaluasi.

        Args:
            splits: Peta nama split ke DataFrame yang sudah punya `text_clean`.

        Returns:
            Tuple (split hasil penyaringan, jumlah baris yang dibuang).
        """
        seen: set[str] = set(splits["train"][TEXT_COLUMN])
        filtered = {"train": splits["train"]}
        removed = 0

        for name in ("val", "test"):
            frame = splits[name]
            kept = frame[~frame[TEXT_COLUMN].isin(seen)].reset_index(drop=True)
            removed += len(frame) - len(kept)
            filtered[name] = kept
            seen |= set(kept[TEXT_COLUMN])

        logger.info(
            "Duplikat text_clean lintas-split dibuang dari val/test: %d "
            "(val %d -> %d, test %d -> %d)",
            removed,
            len(splits["val"]),
            len(filtered["val"]),
            len(splits["test"]),
            len(filtered["test"]),
        )
        return filtered, removed

    def build(self, raw_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Jalankan pipeline lengkap dari DataFrame mentah ke tiga split bersih.

        `clean_text` sengaja dijalankan SETELAH split: pembersihan bersifat
        per-baris sehingga tidak membocorkan informasi antar split, dan urutan
        ini menjaga kunci dedup tetap mengacu pada teks asli.

        Args:
            raw_frame: DataFrame mentah dengan kolom teks dan label.

        Returns:
            Dict {"train": df, "val": df, "test": df} berisi kolom
            `textOriginal`, `text_clean`, `label`, dan `nfkc_key`.

        Raises:
            KeyError: Kalau kolom teks atau label tidak ada.
        """
        self.counts["raw"] = len(raw_frame)

        frame = raw_frame[[RAW_TEXT_COLUMN, LABEL_COLUMN]].dropna()
        self.counts["after_missing"] = len(frame)

        deduped = self.deduplicate(frame)
        self.counts["after_dedup"] = len(deduped)

        train_df, val_df, test_df = self.stratified_split(deduped)
        splits = {"train": train_df, "val": val_df, "test": test_df}

        for split_frame in splits.values():
            split_frame[TEXT_COLUMN] = self.cleaner.clean_series(
                split_frame[RAW_TEXT_COLUMN]
            )

        overlaps = self.find_leakage(splits)
        if any(overlaps.values()):
            raise ValueError(f"kebocoran nfkc_key antar split terdeteksi: {overlaps}")

        splits, removed = self.drop_cross_split_duplicates(splits)
        self.counts["leakage_removed"] = removed
        self.counts["final_total"] = sum(len(df) for df in splits.values())

        residual = self.find_leakage(splits, key=TEXT_COLUMN)
        if any(residual.values()):
            raise ValueError(f"kebocoran text_clean masih tersisa: {residual}")

        return splits

    def write(
        self,
        splits: dict[str, pd.DataFrame],
        output_dir: Path | None = None,
        source: Path | None = None,
        interim_path: Path | None = None,
    ) -> dict[str, Path]:
        """Tulis CSV tiap split, gabungannya, dan `metadata.json`.

        Args:
            splits: Hasil `build`.
            output_dir: Folder tujuan split; `None` memakai `settings.processed_dir`.
            source: Path CSV mentah, dicatat di metadata untuk keterlacakan.
            interim_path: Tujuan gabungan tiga split (`data_clean.csv`);
                `None` memakai `settings.clean_csv`.

        Returns:
            Dict nama artefak ke path yang ditulis.

        Raises:
            OSError: Kalau penulisan gagal.
        """
        output_dir = output_dir or settings.processed_dir
        columns = [RAW_TEXT_COLUMN, TEXT_COLUMN, LABEL_COLUMN]

        written: dict[str, Path] = {}
        for name in SPLIT_NAMES:
            written[name] = write_csv(output_dir / f"{name}.csv", splits[name][columns])

        written["combined"] = write_csv(
            interim_path or settings.clean_csv,
            pd.concat(
                [splits[name][columns].assign(split=name) for name in SPLIT_NAMES],
                ignore_index=True,
            ),
        )

        class_weights = self.compute_class_weights(splits["train"][LABEL_COLUMN])
        written["metadata"] = write_json(
            output_dir / "metadata.json",
            self.metadata(splits, class_weights, source=source),
        )
        logger.info("Split dan metadata ditulis ke %s", output_dir)
        return written

    def metadata(
        self,
        splits: dict[str, pd.DataFrame],
        class_weights: dict[int, float],
        source: Path | None = None,
    ) -> dict[str, object]:
        """Susun metadata dataset yang menyertai split.

        Args:
            splits: Hasil `build`.
            class_weights: Hasil `compute_class_weights` atas split train.
            source: Path CSV mentah.

        Returns:
            Dict metadata siap ditulis sebagai JSON.
        """
        split_stats: dict[str, dict[str, float]] = {}
        for name, frame in splits.items():
            n = len(frame)
            n_positive = int((frame[LABEL_COLUMN] == 1).sum())
            split_stats[name] = {
                "n": n,
                "L0": n - n_positive,
                "L1": n_positive,
                "L1_pct": round(n_positive / n * 100, 2) if n else 0.0,
            }

        return {
            "source": self._relative_source(source),
            "random_seed": self.seed,
            "counts": self.counts,
            "label_conflict": self.label_conflict,
            "splits": split_stats,
            "class_weights": {str(k): v for k, v in class_weights.items()},
            "config": {
                "dedup_key": "nfkc_exact",
                "split_ratios": list(self.ratios),
                "stratified": True,
                "max_length": settings.max_length,
                "model_name": settings.base_model,
                "special_tokens": [
                    URL_PLACEHOLDER,
                    MENTION_PLACEHOLDER,
                    NUM_PLACEHOLDER,
                ],
                "placeholders": {
                    "url": URL_PLACEHOLDER,
                    "mention": MENTION_PLACEHOLDER,
                    "number": f"standalone \\b\\d+\\b -> {NUM_PLACEHOLDER}",
                },
                "kept_as_is": [
                    "short_comments",
                    "non_indonesian",
                    "length_outliers",
                    "emoji",
                ],
                "lowercase_manual": False,
            },
            "notes": (
                "Indeks FAISS RM-c HANYA dari split train. "
                "Model wajib resize_token_embeddings(len(tokenizer))."
            ),
        }

    @staticmethod
    def _relative_source(source: Path | None) -> str | None:
        """Path sumber relatif terhadap root repo, agar metadata tetap portabel."""
        if source is None:
            return None
        try:
            return str(Path(source).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return str(source)
