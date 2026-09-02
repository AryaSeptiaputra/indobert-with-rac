"""Satukan kurva per-epoch yang tersebar SATU FILE PER RUN menjadi satu file per skenario.

    history/rma_run1.csv ... rma_run26.csv  ->  history/rma_history.csv  (+ kolom run_id)

Dipakai sekali untuk memigrasi data lama; run baru sudah langsung menulis ke file gabungan
lewat `append_history` di src/job_runner.py.

    python scripts/merge_history.py                 # results/vast + results/vast_rmc_cheap_head
    python scripts/merge_history.py --dry-run       # cetak rencana saja, nol perubahan
    python scripts/merge_history.py --keep          # gabungkan, file per-run dibiarkan di tempat
    python scripts/merge_history.py results_c2/vast_candidate2   # folder lain

SCRIPT INI TIDAK PERNAH MENGHAPUS FILE. Setelah hasil gabungan diverifikasi ulang dari disk,
file per-run DIPINDAHKAN ke history/_backup_per_run/. Menghapus folder cadangan itu adalah
langkah manual Anda, setelah puas dengan hasilnya. (Penting: results/vast/ di-gitignore, jadi
file di sana tidak ada salinannya di git.)
"""

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [ROOT / "results" / "vast", ROOT / "results" / "vast_rmc_cheap_head"]
BACKUP_DIRNAME = "_backup_per_run"

# rma_run7.csv, dan juga rma_run1_c2.csv hasil rename manual.
RUN_FILE_RE = re.compile(r"^(?P<scenario>.+?)_run(?P<run_id>\d+)")


def collect(hist_dir: Path):
    """{scenario: [(run_id, path), ...]} dari file per-run di `hist_dir` (non-rekursif).

    Glob non-rekursif -> folder cadangan tidak pernah ikut terbaca, jadi script aman
    dijalankan berulang kali.
    """
    groups = defaultdict(list)
    for p in sorted(hist_dir.glob("*_run*.csv")):
        if p.name.endswith("_history.csv"):
            continue
        m = RUN_FILE_RE.match(p.stem)
        if not m:
            print(f"    ! nama tak dikenali, dilewati: {p.name}")
            continue
        groups[m.group("scenario")].append((int(m.group("run_id")), p))
    for sc in groups:
        groups[sc].sort()
    return groups


def merge_group(scenario: str, files, out_path: Path, dry_run: bool):
    """Gabungkan file per-run satu skenario. Return (df, expected) atau (None, None) bila di-skip."""
    header, frames, expected = None, [], {}
    for run_id, p in files:
        d = pd.read_csv(p)
        if header is None:
            header = list(d.columns)
        elif list(d.columns) != header:
            print(f"    ! header {p.name} berbeda dari {files[0][1].name} -- grup '{scenario}' dilewati")
            return None, None
        d.insert(0, "run_id", run_id)
        frames.append(d)
        expected[run_id] = len(d)

    merged = pd.concat(frames, ignore_index=True)

    # Kalau file gabungan sudah ada (mis. sudah ada run baru dari append_history), isinya ikut
    # dipertahankan; run_id yang sama diambil dari file per-run yang sedang dimigrasi.
    if out_path.exists():
        prev = pd.read_csv(out_path)
        if "run_id" in prev.columns:
            keep = prev[~prev["run_id"].isin(merged["run_id"])]
            if len(keep):
                print(f"    + {out_path.name} sudah ada: {len(keep)} baris ({keep['run_id'].nunique()} run) dipertahankan")
                merged = pd.concat([keep, merged], ignore_index=True)

    merged = merged.sort_values("run_id", kind="stable").reset_index(drop=True)
    if not dry_run:
        merged.to_csv(out_path, index=False)
    return merged, expected


def verify(out_path: Path, expected: dict) -> bool:
    """Baca ULANG hasil dari disk dan cocokkan dengan file sumber. Gagal -> jangan pindahkan apa pun."""
    got = pd.read_csv(out_path)
    counts = got["run_id"].value_counts().to_dict()
    ok = True
    for run_id, n in expected.items():
        if counts.get(run_id) != n:
            print(f"    !! run_id {run_id}: {counts.get(run_id)} baris di gabungan, harusnya {n}")
            ok = False
    missing = set(expected) - set(counts)
    if missing:
        print(f"    !! run_id hilang dari gabungan: {sorted(missing)}")
        ok = False
    return ok


def process_dir(d: Path, dry_run: bool, keep: bool) -> bool:
    hist = d / "history"
    print(f"\n=== {d.relative_to(ROOT).as_posix() if d.is_relative_to(ROOT) else d} ===")
    if not hist.is_dir():
        print("    (tidak ada folder history/) -- dilewati")
        return True

    groups = collect(hist)
    if not groups:
        print("    tidak ada file per-run tersisa -- tidak ada yang perlu digabung")
        return True

    all_ok = True
    for scenario, files in groups.items():
        out_path = hist / f"{scenario}_history.csv"
        n_rows = sum(len(pd.read_csv(p)) for _, p in files)
        print(f"  [{scenario}] {len(files)} file, {n_rows} baris data -> {out_path.name}")

        merged, expected = merge_group(scenario, files, out_path, dry_run)
        if merged is None:
            all_ok = False
            continue
        if dry_run:
            print(f"    (dry run) akan menulis {len(merged)} baris, run_id "
                  f"{files[0][0]}-{files[-1][0]}, lalu memindahkan {len(files)} file ke {BACKUP_DIRNAME}/")
            continue

        if not verify(out_path, expected):
            print(f"    !! VERIFIKASI GAGAL -- tidak ada file yang dipindahkan untuk '{scenario}'")
            all_ok = False
            continue
        print(f"    verifikasi OK: {len(merged)} baris, {merged['run_id'].nunique()} run")

        if keep:
            print(f"    --keep: {len(files)} file per-run dibiarkan di tempatnya")
            continue
        backup = hist / BACKUP_DIRNAME
        backup.mkdir(exist_ok=True)
        for _, p in files:
            shutil.move(str(p), str(backup / p.name))
        print(f"    {len(files)} file per-run dipindahkan ke {backup.relative_to(d).as_posix()}/ (tidak dihapus)")
    return all_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="out_dir yang diproses (default: results/vast, results/vast_rmc_cheap_head)")
    ap.add_argument("--dry-run", action="store_true", help="cetak rencana saja, tidak mengubah apa pun")
    ap.add_argument("--keep", action="store_true", help="jangan pindahkan file per-run setelah digabung")
    args = ap.parse_args(argv)

    dirs = [Path(x).resolve() for x in args.dirs] if args.dirs else DEFAULT_DIRS
    if args.dry_run:
        print("[DRY RUN -- tidak ada file yang ditulis, dipindah, atau dihapus]")

    ok = all([process_dir(d, args.dry_run, args.keep) for d in dirs])

    if args.dry_run:
        print("\nDry run selesai.")
    elif ok and not args.keep:
        print(f"\nSelesai. File per-run ada di history/{BACKUP_DIRNAME}/ -- silakan hapus manual "
              f"setelah Anda cek sendiri hasil gabungannya.")
    elif not ok:
        print("\nSelesai DENGAN MASALAH -- lihat baris '!!' di atas. File sumber tetap utuh.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
