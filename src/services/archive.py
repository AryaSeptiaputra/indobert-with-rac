"""Arsip hasil kampanye ke satu berkas `.tar.gz` yang bisa diunduh dari instance.

`outputs/` tidak disimpan di git: `runs_*.csv` dan `best.json` yang ikut ter-clone
membuat kampanye baru melewati semua konfigurasi dan tanpa checkpoint juara. Akibatnya
hasil hanya ada di disk mesin tempat kampanye berjalan, dan hilang bersama instance
Vast.ai yang dihancurkan. Angka efisiensi (waktu latih, latency, memori) tidak bisa
dibuat ulang karena hanya sah dari sesi aslinya, jadi hasil itu harus diamankan.

Secara bawaan checkpoint dan cache fitur tidak ikut diarsipkan: keduanya besar
(ratusan MB) dan bisa dibangun ulang (`CampaignRunner.restore_checkpoints`), sedangkan
berkas hasil yang tidak tergantikan hanya beberapa MB.
"""

from __future__ import annotations

import os
import re
import tarfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.utils.io import read_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Folder besar yang bisa dibangun ulang dan karena itu tidak ikut diarsipkan.
HEAVY_DIRS = ("checkpoints", "features")

NO_HARDWARE_LABEL = "tanpa-hardware"

# Deretan karakter selain huruf kecil dan angka diganti satu tanda hubung, sehingga
# "NVIDIA GeForce RTX 3090" menjadi "nvidia-geforce-rtx-3090".
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


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
