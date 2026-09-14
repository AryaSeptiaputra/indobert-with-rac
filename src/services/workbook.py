"""Ekspor hasil kampanye ke satu berkas Excel multi-sheet.

Dipakai untuk menyiapkan bahan Bab 4: seluruh riwayat run, tabel perbandingan
final, verdict kriteria sukses, dan pivot grid dikumpulkan ke satu berkas yang
bisa dibuka tanpa Python.

Isinya sepenuhnya diturunkan dari artefak di folder keluaran, tanpa satu pun
angka yang ditulis tangan. Versi sebelumnya menanamkan narasi dan angka hasil
kampanye tertentu langsung di kode, sehingga berkasnya berbohong begitu
kampanye diulang.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import SCENARIOS
from src.utils.io import read_csv, read_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Excel membatasi nama sheet pada 31 karakter.
MAX_SHEET_NAME = 31

SCENARIO_LABELS = {
    "rma": "RM-a full fine-tuning",
    "rmb": "RM-b frozen encoder",
    "rmc": "RM-c frozen + RAC",
}


class WorkbookBuilder:
    """Kumpulkan artefak satu kampanye menjadi satu workbook Excel.

    Args:
        out_dir: Folder keluaran kampanye yang akan diekspor.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)

    def sheets(self) -> dict[str, pd.DataFrame]:
        """Susun seluruh sheet yang tersedia.

        Sheet yang sumbernya belum ada dilewati, sehingga workbook tetap bisa
        dibuat di tengah kampanye.

        Returns:
            Peta nama sheet ke DataFrame-nya, dimulai dari sheet ringkasan.
        """
        sheets: dict[str, pd.DataFrame] = {"Ringkasan": self.summary_sheet()}

        for scenario in SCENARIOS:
            runs = read_csv(self.out_dir / f"runs_{scenario}.csv")
            if not runs.empty:
                sheets[f"Run {scenario.upper()}"] = runs

            errors = read_csv(self.out_dir / f"runs_{scenario}_errors.csv")
            if not errors.empty:
                sheets[f"Galat {scenario.upper()}"] = errors

            history = read_csv(self.out_dir / "history" / f"{scenario}_history.csv")
            if not history.empty:
                sheets[f"Kurva {scenario.upper()}"] = history

        for name, filename in (
            ("Perbandingan Final", "final_comparison.csv"),
            ("Kriteria Sukses", "success_criteria.csv"),
            ("Benchmark Inferensi", "inference_benchmark.csv"),
        ):
            frame = read_csv(self.out_dir / "metrics" / filename)
            if not frame.empty:
                sheets[name] = frame

        for pivot_path in sorted((self.out_dir / "metrics").glob("*_grid_pivot_*.csv")):
            frame = read_csv(pivot_path)
            if not frame.empty:
                sheets[self._sheet_name(f"Pivot {pivot_path.stem}")] = frame

        return sheets

    def summary_sheet(self) -> pd.DataFrame:
        """Sheet ringkasan: jumlah run, juara, dan konteks hardware.

        Returns:
            DataFrame dua kolom (`keterangan`, `nilai`).
        """
        rows: list[dict[str, object]] = []

        hardware = read_json(self.out_dir / "hardware.json", default={}) or {}
        for key in ("gpu", "vram_total_mb", "torch", "transformers", "cuda_version", "recorded_at"):
            if key in hardware:
                rows.append({"keterangan": f"hardware.{key}", "nilai": hardware[key]})

        summary = read_json(self.out_dir / "tuning_summary.json", default={}) or {}
        for scenario, entry in (summary.get("scenarios") or {}).items():
            label = SCENARIO_LABELS.get(scenario, scenario)
            for key, value in entry.items():
                rows.append({"keterangan": f"{label} | {key}", "nilai": value})

        best = read_json(self.out_dir / "best.json", default={}) or {}
        for scenario, entry in best.items():
            label = SCENARIO_LABELS.get(scenario, scenario)
            rows.append({"keterangan": f"{label} | config juara", "nilai": str(entry.get("config"))})

        if not rows:
            rows.append({"keterangan": "status", "nilai": "belum ada artefak kampanye"})

        return pd.DataFrame(rows)

    def build(self, path: str | Path | None = None) -> Path:
        """Tulis workbook Excel.

        Args:
            path: Lokasi berkas tujuan; `None` menulis `HASIL.xlsx` di folder
                keluaran kampanye.

        Returns:
            Path berkas yang ditulis.

        Raises:
            OSError: Kalau penulisan gagal.
            ValueError: Kalau tidak ada satu pun sheet yang bisa dibuat.
        """
        sheets = self.sheets()
        if not sheets:
            raise ValueError(f"tidak ada artefak yang bisa diekspor dari {self.out_dir}")

        target = Path(path) if path else self.out_dir / "HASIL.xlsx"
        target.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=self._sheet_name(name), index=False)
                self._autosize(writer.sheets[self._sheet_name(name)], frame)

        logger.info("Workbook ditulis ke %s (%d sheet)", target, len(sheets))
        return target

    @staticmethod
    def _sheet_name(name: str) -> str:
        return name[:MAX_SHEET_NAME]

    @staticmethod
    def _autosize(worksheet: object, frame: pd.DataFrame, max_width: int = 60) -> None:
        from openpyxl.utils import get_column_letter

        for index, column in enumerate(frame.columns, start=1):
            widest = max(
                [len(str(column))]
                + [len(str(value)) for value in frame[column].head(200)]
            )
            worksheet.column_dimensions[get_column_letter(index)].width = min(
                widest + 2, max_width
            )
