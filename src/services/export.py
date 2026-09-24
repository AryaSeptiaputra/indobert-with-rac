"""Ekspor hasil kampanye: workbook Excel, gabungan riwayat run, dan arsip `.tar.gz`.

`WorkbookBuilder` mengumpulkan hasil kampanye ke satu berkas Excel multi-sheet.

Dipakai untuk menyiapkan bahan Bab 4: seluruh riwayat run, tabel perbandingan
final, verdict kriteria sukses, dan pivot grid dikumpulkan ke satu berkas yang
bisa dibuka tanpa Python.

Isinya sepenuhnya diturunkan dari artefak di folder keluaran, tanpa satu pun
angka yang ditulis tangan. Versi sebelumnya menanamkan narasi dan angka hasil
kampanye tertentu langsung di kode, sehingga berkasnya berbohong begitu
kampanye diulang.

`RunMerger` menggabungkan riwayat run dari beberapa folder kampanye, misalnya
kampanye utama ditambah eksplorasi terpisah dengan encoder berbeda. Kunci baris
hasil gabungan adalah pasangan (`source`, `run_id`) karena tiap folder memulai
penomoran dari 1, dan kolom `delta_vs_best_f1_macro_pp` serta `is_tie_with_best`
hanya bermakna di dalam satu `source`. Kolom waktu, memori, dan latency TIDAK
boleh dibandingkan lintas `source`: angka efisiensi hanya sah bila berasal dari
satu hardware dan satu sesi. F1 tidak terpengaruh hardware sehingga aman
dibandingkan. Berkas sumber tidak pernah diubah; menjalankan ulang hanya menulis
ulang keluaran.

`ResultArchiver` membungkus hasil kampanye ke satu berkas `.tar.gz` yang bisa
diunduh dari instance. `outputs/` tidak disimpan di git: `runs_*.csv` dan
`best.json` yang ikut ter-clone membuat kampanye baru melewati semua konfigurasi
dan tanpa checkpoint juara. Akibatnya hasil hanya ada di disk mesin tempat
kampanye berjalan, dan hilang bersama instance Vast.ai yang dihancurkan. Angka
efisiensi (waktu latih, latency, memori) tidak bisa dibuat ulang karena hanya sah
dari sesi aslinya, jadi hasil itu harus diamankan. Secara bawaan checkpoint dan
cache fitur tidak ikut diarsipkan: keduanya besar (ratusan MB) dan bisa dibangun
ulang (`CampaignRunner.restore_checkpoints`), sedangkan berkas hasil yang tidak
tergantikan hanya beberapa MB.
"""

from __future__ import annotations

import os
import re
import tarfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import SCENARIOS
from src.utils.io import read_csv, read_json, write_csv
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Excel membatasi nama sheet pada 31 karakter.
MAX_SHEET_NAME = 31

SCENARIO_LABELS = {
    "rma": "RM-a full fine-tuning",
    "rmb": "RM-b frozen encoder",
    "rmc": "RM-c frozen + RAC",
}

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

# Folder besar yang bisa dibangun ulang dan karena itu tidak ikut diarsipkan.
HEAVY_DIRS = ("checkpoints", "features")

NO_HARDWARE_LABEL = "tanpa-hardware"

# Deretan karakter selain huruf kecil dan angka diganti satu tanda hubung, sehingga
# "NVIDIA GeForce RTX 3090" menjadi "nvidia-geforce-rtx-3090".
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


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


@dataclass(frozen=True)
class ArchiveResult:
    """Hasil pembuatan arsip.

    Attributes:
        path: Lokasi berkas arsip.
        size_bytes: Ukuran arsip terkompresi.
        n_files: Jumlah berkas di dalam arsip.
    """

    path: Path
    size_bytes: int
    n_files: int


class ResultArchiver:
    """Bungkus folder hasil (dan berkas tambahan) menjadi satu `.tar.gz`.

    Args:
        source_dir: Folder hasil yang diarsipkan, biasanya `outputs/`.
        destination_dir: Folder tempat arsip ditulis; tidak boleh berada di dalam
            `source_dir`.
        include_heavy: Ikutkan folder `checkpoints/` dan `features/`.

    Raises:
        ValueError: Kalau `destination_dir` berada di dalam `source_dir`.
    """

    def __init__(
        self,
        source_dir: Path,
        destination_dir: Path,
        include_heavy: bool = False,
    ) -> None:
        self.source_dir = Path(source_dir).resolve()
        self.destination_dir = Path(destination_dir).resolve()
        self.exclude_dirs: tuple[str, ...] = () if include_heavy else HEAVY_DIRS

        if self.destination_dir == self.source_dir or self.source_dir in self.destination_dir.parents:
            raise ValueError(
                f"folder tujuan {self.destination_dir} tidak boleh berada di dalam {self.source_dir}"
            )

    def _hardware_label(self) -> str:
        for path in sorted(self.source_dir.glob("*/hardware.json")):
            gpu = str((read_json(path, default={}) or {}).get("gpu", ""))
            slug = _NON_SLUG_RE.sub("-", gpu.lower()).strip("-")
            if slug:
                return slug
        return NO_HARDWARE_LABEL

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []
        for root, dirs, names in os.walk(self.source_dir):
            dirs[:] = sorted(name for name in dirs if name not in self.exclude_dirs)
            files.extend(Path(root) / name for name in sorted(names))
        return files

    def archive(self, extra_files: Sequence[Path] = ()) -> ArchiveResult:
        """Buat arsip `hasil_<gpu>_<waktu>.tar.gz` di folder tujuan.

        Args:
            extra_files: Berkas tambahan di luar `source_dir`, misalnya `HASIL.xlsx`.
                Yang belum ada dilewati.

        Returns:
            Lokasi, ukuran, dan jumlah berkas arsip.

        Raises:
            FileNotFoundError: Kalau `source_dir` tidak ada atau tidak berisi berkas apa pun.
            OSError: Kalau penulisan arsip gagal.
        """
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"folder hasil tidak ditemukan: {self.source_dir}")

        files = self._collect_files()
        extras = [Path(path) for path in extra_files if Path(path).is_file()]
        if not files and not extras:
            raise FileNotFoundError(f"tidak ada hasil untuk diarsipkan di {self.source_dir}")

        self.destination_dir.mkdir(parents=True, exist_ok=True)
        # Contoh: hasil_nvidia-geforce-rtx-3090_20260920-143005.tar.gz
        name = f"hasil_{self._hardware_label()}_{time.strftime('%Y%m%d-%H%M%S')}.tar.gz"
        target = self.destination_dir / name
        partial = target.with_name(target.name + ".tmp")

        try:
            with tarfile.open(partial, "w:gz") as archive:
                for path in files:
                    arcname = Path(self.source_dir.name) / path.relative_to(self.source_dir)
                    archive.add(path, arcname=arcname.as_posix(), recursive=False)
                for path in extras:
                    archive.add(path, arcname=path.name, recursive=False)
            os.replace(partial, target)
        except OSError:
            partial.unlink(missing_ok=True)
            logger.error("Pembuatan arsip %s gagal", target, exc_info=True)
            raise

        result = ArchiveResult(target, target.stat().st_size, len(files) + len(extras))
        logger.info("Arsip hasil ditulis: %s (%d berkas)", result.path, result.n_files)
        return result
