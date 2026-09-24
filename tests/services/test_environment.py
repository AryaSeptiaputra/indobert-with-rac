"""Test gate lingkungan satu-hardware (Aturan #1, Tabel 4.1)."""

from __future__ import annotations

import json

import pytest

from src.services import environment
from src.services.environment import (
    EnvironmentMismatchError,
    record_tuning_session,
    verify_final_session,
)

LINGKUNGAN = {
    "gpu": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576, "driver": "550.54",
    "cuda_version": "12.4", "cpu": "AMD EPYC", "cpu_logical_cores": 32, "ram_total_gb": 125.7,
    "os": "Linux", "python": "3.13.1", "torch": "2.12.1", "transformers": "5.12.1",
    "faiss": "1.14.3", "seed": 42,
}


@pytest.fixture
def mesin(monkeypatch):
    state = {"env": dict(LINGKUNGAN), "boot": "2026-09-24 01:00:00"}
    monkeypatch.setattr(environment, "collect_environment", lambda: dict(state["env"]))
    monkeypatch.setattr(
        environment, "current_session",
        lambda: {"recorded_at": "2026-09-24 02:00:00", "boot_time": state["boot"]},
    )
    return state


class TestSesiTuning:
    def test_lingkungan_dan_sesi_tercatat(self, mesin, tmp_path) -> None:
        payload = record_tuning_session(tmp_path / "hardware.json")
        assert payload["gpu"] == LINGKUNGAN["gpu"]
        assert payload["faiss"] == "1.14.3" and payload["seed"] == 42
        assert len(payload["tuning_sessions"]) == 1

    def test_sesi_baru_ditumpuk_bukan_ditimpa(self, mesin, tmp_path) -> None:
        record_tuning_session(tmp_path / "hardware.json")
        record_tuning_session(tmp_path / "hardware.json")
        mesin["boot"] = "2026-09-25 08:00:00"
        payload = record_tuning_session(tmp_path / "hardware.json")
        assert [s["boot_time"] for s in payload["tuning_sessions"]] == [
            "2026-09-24 01:00:00", "2026-09-25 08:00:00",
        ]

    def test_lingkungan_lain_ditolak(self, mesin, tmp_path) -> None:
        record_tuning_session(tmp_path / "hardware.json")
        mesin["env"]["gpu"] = "NVIDIA GeForce RTX 3050 Laptop GPU"
        with pytest.raises(EnvironmentMismatchError, match="gpu"):
            record_tuning_session(tmp_path / "hardware.json")


class TestGateFinal:
    def test_lingkungan_identik_sesi_sama(self, mesin, tmp_path) -> None:
        record_tuning_session(tmp_path / "hardware.json")
        payload = verify_final_session(tmp_path / "hardware.json")
        assert payload["final_session"]["same_boot_as_tuning"] is True

    def test_boot_berbeda_hanya_peringatan_dan_tercatat(self, mesin, tmp_path) -> None:
        record_tuning_session(tmp_path / "hardware.json")
        mesin["boot"] = "2026-09-26 09:00:00"
        payload = verify_final_session(tmp_path / "hardware.json")
        saved = json.loads((tmp_path / "hardware.json").read_text(encoding="utf-8"))
        assert payload["final_session"]["same_boot_as_tuning"] is False
        assert saved["final_session"]["boot_time"] == "2026-09-26 09:00:00"
        assert saved["tuning_sessions"][0]["boot_time"] == "2026-09-24 01:00:00"

    @pytest.mark.parametrize("key", ["gpu", "vram_total_mb", "driver", "cuda_version", "torch", "transformers", "faiss"])
    def test_kunci_ketat_berbeda_menghentikan(self, mesin, tmp_path, key) -> None:
        record_tuning_session(tmp_path / "hardware.json")
        mesin["env"][key] = "lain"
        with pytest.raises(EnvironmentMismatchError, match=key):
            verify_final_session(tmp_path / "hardware.json")

    def test_tanpa_catatan_tuning_ditolak(self, mesin, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="03a"):
            verify_final_session(tmp_path / "hardware.json")


def test_lingkungan_sungguhan_lengkap() -> None:
    env = environment.collect_environment()
    assert set(env) >= {"gpu", "cpu", "ram_total_gb", "os", "python", "torch", "faiss", "seed"}
