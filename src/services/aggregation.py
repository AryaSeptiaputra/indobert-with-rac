"""Penggabungan riwayat run dari beberapa folder kampanye.

Berguna saat satu skenario dijalankan di lebih dari satu folder keluaran,
misalnya kampanye utama ditambah eksplorasi terpisah dengan encoder berbeda.

Peringatan penting saat memakai hasil gabungan: kunci baris adalah pasangan
(`source`, `run_id`) karena tiap folder memulai penomoran dari 1, dan kolom
`delta_vs_best_f1_macro_pp` serta `is_tie_with_best` hanya bermakna di dalam
satu `source`. Kolom waktu, memori, dan latency TIDAK boleh dibandingkan
lintas `source`: angka efisiensi hanya sah bila berasal dari satu hardware dan
satu sesi. F1 tidak terpengaruh hardware sehingga aman dibandingkan.

Berkas sumber tidak pernah diubah; menjalankan ulang hanya menulis ulang
keluaran.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import SCENARIOS
from src.utils.io import read_csv, write_csv
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

LEAD_COLUMNS = ("source", "run_id", "scenario")

MERGE_README = """# outputs/combined -- turunan, bukan sumber kebenaran

Dihasilkan oleh `src.services.aggregation.RunMerger`. Berkas sumber tidak pernah
diubah; jalankan ulang untuk menyegarkan.

Aturan membaca:

- Kunci baris adalah pasangan (`source`, `run_id`). Tiap folder kampanye memulai
  penomoran run dari 1, jadi `run_id` sendirian tidak unik.
- `delta_vs_best_f1_macro_pp` dan `is_tie_with_best` dihitung relatif terhadap
  juara di dalam satu folder, sehingga hanya bermakna di dalam satu `source`.
- Kolom waktu, memori, dan latency **tidak boleh** dibandingkan lintas `source`:
  angka efisiensi hanya sah bila diukur pada satu hardware dan satu sesi.
- F1 bersifat hardware-independent dan aman dibandingkan lintas `source`.
"""


class RunMerger:
    """Gabungkan `runs_{scenario}.csv` dari beberapa folder kampanye.

    Args:
        sources: Peta nama sumber ke folder kampanye-nya. Nama itulah yang
            muncul di kolom `source`.
    """

    def __init__(self, sources: dict[str, str | Path]) -> None:
        self.sources = {name: Path(path) for name, path in sources.items()}

    def merge_scenario(self, scenario: str) -> pd.DataFrame:
        """Gabungkan riwayat satu skenario dari seluruh sumber.

        Args:
            scenario: Kode skenario.

        Returns:
            DataFrame gabungan dengan kolom `source` di depan; kosong bila tidak
            ada satu pun sumber yang memuat skenario itu.
        """
        frames: list[pd.DataFrame] = []

        for name, directory in self.sources.items():
            path = directory / f"runs_{scenario}.csv"
            frame = read_csv(path)
            if frame.empty:
                logger.debug("Sumber %s tidak punya %s", name, path.name)
                continue
            frame = frame.copy()
            frame.insert(0, "source", name)
            frames.append(frame)
            logger.info("Sumber %s: %d run %s", name, len(frame), scenario)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        return merged[self._ordered_columns(merged)]

    def merge_all(self, out_dir: str | Path) -> dict[str, Path]:
        """Gabungkan ketiga skenario dan tulis hasilnya beserta README.

        Args:
            out_dir: Folder tujuan berkas gabungan.

        Returns:
            Peta skenario ke path berkas yang ditulis, plus kunci "readme".

        Raises:
            OSError: Kalau penulisan gagal.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        written: dict[str, Path] = {}
        for scenario in SCENARIOS:
            merged = self.merge_scenario(scenario)
            if merged.empty:
                continue
            written[scenario] = write_csv(out_dir / f"runs_{scenario}.csv", merged)
            logger.info("Gabungan %s: %d baris", scenario, len(merged))

        readme = out_dir / "README.md"
        readme.write_text(MERGE_README, encoding="utf-8")
        written["readme"] = readme
        return written

    @staticmethod
    def _ordered_columns(frame: pd.DataFrame) -> list[str]:
        lead = [column for column in LEAD_COLUMNS if column in frame.columns]
        rest = [column for column in frame.columns if column not in lead]
        return lead + rest
