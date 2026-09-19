"""Pencatatan riwayat run: baris hasil, kurva per-epoch, dan juara sejauh ini.

Riwayat run MENUMPUK lintas sesi dan tidak pernah ditimpa. Setiap run menempati
satu baris berisi seluruh hyperparameter, metrik validation, biaya komputasi,
dan kolom `catatan` yang merekam alasan konfigurasi itu dicoba. Kolom terakhir
itulah yang membuat tiap nilai hyperparameter punya justifikasi eksplisit saat
ditulis di Bab 4.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.config import settings
from src.utils.io import CorruptArtifactError, read_csv, read_json, write_csv, write_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class RunLogger:
    """Riwayat run satu skenario, satu baris per konfigurasi.

    Args:
        path: Lokasi `runs_{scenario}.csv`.
        tie_threshold_pp: Selisih F1-macro (poin persentase) yang masih dianggap
            seri; `None` memakai konfigurasi.

    Raises:
        CorruptArtifactError: Kalau berkas riwayat ada tapi tidak bisa diurai.
            Sengaja dilempar alih-alih dilanjutkan dengan riwayat kosong: versi
            sebelumnya menelan galat ini lalu menulis ulang berkas dari daftar
            kosong, sehingga satu CSV yang sedang terbuka di Excel bisa
            menghapus puluhan run sekaligus.
    """

    def __init__(self, path: str | Path, tie_threshold_pp: float | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tie_threshold_pp = (
            settings.tie_threshold_pp if tie_threshold_pp is None else tie_threshold_pp
        )

        frame = read_csv(self.path)
        self.rows: list[dict[str, object]] = frame.to_dict("records")
        if self.rows:
            logger.info("Riwayat %s dimuat: %d run", self.path.name, len(self.rows))

    def next_id(self) -> int:
        """Nomor run berikutnya (1-indeks)."""
        return len(self.rows) + 1

    def best_f1_macro(self) -> float:
        """F1-macro validation tertinggi yang tercatat sejauh ini; -1 bila kosong."""
        values = [
            float(row["val_f1_macro"])
            for row in self.rows
            if row.get("val_f1_macro") is not None
        ]
        return max(values) if values else -1.0

    def log(self, row: dict[str, object]) -> dict[str, object]:
        """Tambahkan satu baris hasil run dan simpan ke disk.

        Kolom turunan dihitung di sini agar analisis lintas-run tidak perlu
        menghitung ulang: `delta_vs_best_f1_macro_pp` dan `is_tie_with_best`
        mengoperasionalkan aturan seleksi (selisih di bawah ambang dianggap seri
        sehingga konfigurasi yang lebih murah menang), sedangkan
        `overfit_signal` menandai run yang epoch terbaiknya bukan epoch terakhir.

        Args:
            row: Kolom hasil run; minimal memuat `val_f1_macro`.

        Returns:
            Baris final termasuk `run_id`, kolom turunan, dan `timestamp`.

        Raises:
            OSError: Kalau penulisan CSV gagal.
        """
        enriched = dict(row)

        prior_best = self.best_f1_macro()
        if "val_f1_macro" in enriched and prior_best > -1:
            delta_pp = (float(enriched["val_f1_macro"]) - prior_best) * 100
            enriched["delta_vs_best_f1_macro_pp"] = round(delta_pp, 4)
            enriched["is_tie_with_best"] = bool(abs(delta_pp) <= self.tie_threshold_pp)

        if "best_epoch" in enriched and "epochs" in enriched:
            enriched["overfit_signal"] = bool(
                int(enriched["best_epoch"]) < int(enriched["epochs"])
            )

        final = {
            "run_id": self.next_id(),
            **enriched,
            "timestamp": time.strftime(TIMESTAMP_FORMAT),
        }
        self.rows.append(final)
        write_csv(self.path, pd.DataFrame(self.rows))
        return final

    def log_error(
        self,
        scenario: str,
        batch_id: str,
        sequence: int,
        config: dict[str, object],
        note: str,
        error: Exception,
    ) -> Path:
        """Catat konfigurasi yang gagal ke berkas galat terpisah.

        Kegagalan satu konfigurasi tidak boleh membatalkan sisa batch, tapi juga
        tidak boleh hilang tanpa jejak.

        Args:
            scenario: Kode skenario.
            batch_id: Penanda batch asal.
            sequence: Nomor urut konfigurasi di dalam batch.
            config: Hyperparameter yang gagal.
            note: Catatan konfigurasi itu.
            error: Exception yang terjadi.

        Returns:
            Path berkas galat yang ditulis.
        """
        path = self.path.parent / f"runs_{scenario}_errors.csv"
        entry = {
            "batch_id": batch_id,
            "seq": sequence,
            "error_type": type(error).__name__,
            "error": str(error)[:500],
            "note": note,
            "timestamp": time.strftime(TIMESTAMP_FORMAT),
            **config,
        }
        existing = read_csv(path)
        combined = pd.concat([existing, pd.DataFrame([entry])], ignore_index=True)
        write_csv(path, combined)
        logger.warning("Konfigurasi gagal dicatat ke %s: %s", path.name, error)
        return path


class HistoryWriter:
    """Kurva validation per-epoch, satu berkas menumpuk per skenario.

    Satu berkas per run akan berarti 26 berkas terpisah untuk RM-a saja dan
    menyulitkan perbandingan lintas-run, jadi seluruh run masuk ke satu berkas
    dengan kolom `run_id` sebagai penanda.

    Args:
        out_dir: Folder keluaran kampanye.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.dir = Path(out_dir) / "history"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, scenario: str) -> Path:
        """Lokasi berkas history satu skenario."""
        return self.dir / f"{scenario}_history.csv"

    def append(
        self,
        scenario: str,
        run_id: int,
        history: list[dict[str, float | int]],
    ) -> Path:
        """Tambahkan kurva satu run, menimpa bila `run_id` itu sudah ada.

        Idempoten karena `run_id` diturunkan dari jumlah baris riwayat run;
        kalau berkas riwayat pernah di-reset, nomor bisa terulang dan tanpa
        penimpaan berkas history akan berisi dua kurva dengan id yang sama.

        Args:
            scenario: Kode skenario.
            run_id: Nomor run pemilik kurva.
            history: Daftar baris per-epoch.

        Returns:
            Path berkas history.

        Raises:
            OSError: Kalau penulisan gagal.
        """
        path = self.path(scenario)
        frame = pd.DataFrame(history)
        if frame.empty:
            return path

        frame.insert(0, "run_id", run_id)
        existing = read_csv(path)
        if not existing.empty and "run_id" in existing.columns:
            existing = existing[existing["run_id"] != run_id]
        if not existing.empty:
            frame = pd.concat([existing, frame], ignore_index=True)

        write_csv(path, frame)
        return path


class BestTracker:
    """Konfigurasi terbaik sejauh ini per skenario, diukur dengan F1-macro validation.

    Args:
        out_dir: Folder keluaran kampanye.

    Raises:
        CorruptArtifactError: Kalau `best.json` ada tapi rusak. Versi sebelumnya
            menganggapnya kosong lalu langsung menimpanya, menghapus catatan
            model terbaik.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.path = Path(out_dir) / "best.json"
        self.data: dict[str, dict[str, object]] = read_json(self.path, default={}) or {}

    def get(self, scenario: str) -> dict[str, object]:
        """Catatan juara satu skenario; dict kosong bila belum ada."""
        return self.data.get(scenario, {})

    def update(self, scenario: str, payload: dict[str, object]) -> bool:
        """Perbarui juara bila run ini mengungguli catatan sebelumnya.

        Args:
            scenario: Kode skenario.
            payload: Ringkasan run, minimal memuat `val_f1_macro`.

        Returns:
            True bila juara diperbarui.

        Raises:
            KeyError: Kalau `payload` tidak memuat `val_f1_macro`.
            OSError: Kalau penulisan gagal.
        """
        challenger = float(payload["val_f1_macro"])
        incumbent = float(self.get(scenario).get("val_f1_macro", -1))

        if challenger <= incumbent:
            return False

        self.data[scenario] = payload
        write_json(self.path, self.data)
        logger.info(
            "Juara baru untuk %s: val F1-macro %.4f (sebelumnya %.4f)",
            scenario,
            challenger,
            incumbent,
        )
        return True

    def replace(self, scenario: str, payload: dict[str, object]) -> None:
        """Ganti juara satu skenario tanpa membandingkan F1-macro.

        Untuk keputusan yang dibuat dengan aturan lain daripada `update`, misalnya
        penantang yang harus lolos ambang seri dan bootstrap sebelum boleh
        menggantikan juara.

        Args:
            scenario: Kode skenario.
            payload: Ringkasan juara baru, minimal memuat `val_f1_macro`.

        Raises:
            KeyError: Kalau `payload` tidak memuat `val_f1_macro`.
            OSError: Kalau penulisan gagal.
        """
        if "val_f1_macro" not in payload:
            raise KeyError("payload harus memuat 'val_f1_macro'")

        self.data[scenario] = payload
        write_json(self.path, self.data)
        logger.info(
            "Juara %s diganti: val F1-macro %.4f", scenario, float(payload["val_f1_macro"])
        )


__all__ = ["RunLogger", "HistoryWriter", "BestTracker", "CorruptArtifactError"]
