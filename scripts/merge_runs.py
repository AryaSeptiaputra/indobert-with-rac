"""Satukan CSV ringkasan run (level run: 1 baris = 1 run) dari SEMUA folder eksperimen
menjadi TIGA file: results/combined/runs_{rma,rmb,rmc}.csv.

Latar: tiap campaign menulis ke out_dir-nya sendiri (results/vast, results/vast_rmc_cheap_head,
results_c2/vast_candidate2), sehingga data satu skenario tersebar di beberapa file. Script ini
MURNI ADITIF -- file sumber tidak pernah dibaca-tulis, hanya dibaca.

    python scripts/merge_runs.py             # tulis results/combined/
    python scripts/merge_runs.py --dry-run   # cuma cetak rencana, nol perubahan

CAVEAT yang melekat pada file gabungan (juga ditulis ke results/combined/README.md):

* Kunci baris = pasangan (source, run_id), BUKAN run_id saja. Tiap out_dir memulai penomoran
  dari 1, jadi run_id berulang antar-source. Penomoran ulang global sengaja tidak dilakukan
  supaya tiap baris tetap bisa ditelusuri balik ke figure aslinya (figures/rma_run13_curve.png).
* Kolom delta_vs_best_f1_macro_pp & is_tie_with_best dihitung RunLogger.log (src/tuning.py:74)
  relatif terhadap juara DI DALAM file asalnya -> hanya valid dibaca per-source, jangan dipakai
  lintas-source dan jangan dihitung ulang secara global.
* Aturan validitas Bab 4 tetap berlaku: kolom waktu/memori/latensi hanya sebanding dalam satu
  hardware+sesi. source berbeda = mesin/sesi berbeda -> jangan dibandingkan. F1 aman.
* Angka final Bab 4 tetap bersumber dari results/vast/. Folder combined adalah alat analisis,
  bukan pengganti.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Urutan di sini menentukan urutan baris & urutan kolom pada file gabungan.
SOURCES = [
    ("vast", ROOT / "results" / "vast"),
    ("cheap_head", ROOT / "results" / "vast_rmc_cheap_head"),
    ("candidate2", ROOT / "results_c2" / "vast_candidate2"),
]

SCENARIOS = ["rma", "rmb", "rmc"]

# Campaign results/vast berjalan sebelum kolom model_name ada (ditambahkan bersama infra
# lite-encoder). Encoder-nya terdokumentasi di CLAUDE.md -> diisi eksplisit supaya kosong tidak
# rancu dengan "tidak diketahui" saat membandingkan base vs lite antar-source.
MODEL_NAME_BACKFILL = {"vast": "indobenchmark/indobert-base-p2"}

# Kolom kunci selalu di depan; sisanya menyusul sesuai urutan kemunculan pertama.
LEAD_COLS = ["source", "run_id", "scenario", "model_name"]

README = """# results/combined/

File turunan (bukan sumber kebenaran) hasil `python scripts/merge_runs.py`.
Menggabungkan CSV ringkasan run dari semua folder eksperimen menjadi tiga file:
`runs_rma.csv`, `runs_rmb.csv`, `runs_rmc.csv`.

Sumber yang digabung (nilai kolom `source`):

| `source` | folder asal |
|---|---|
| `vast` | `results/vast/` — campaign utama, angka final Bab 4 |
| `cheap_head` | `results/vast_rmc_cheap_head/` — eksplorasi RM-c dengan head RM-b termurah (linear/256) |
| `candidate2` | `results_c2/vast_candidate2/` — kandidat #2: config runner-up (RM-a wd=0.10, RM-b wd=0.01, RM-c α=0.2/k=3) |

Ketiganya memakai encoder yang sama (`indobenchmark/indobert-base-p2`); yang berbeda adalah
hyperparameter dan head, bukan encoder-nya.

## Cara membaca — baca ini sebelum menganalisis

- **Kunci baris adalah `(source, run_id)`, bukan `run_id` saja.** Tiap folder memulai penomoran
  dari 1, jadi `run_id` berulang antar-source. Penomoran ulang global sengaja tidak dilakukan agar
  tiap baris tetap bisa ditelusuri balik ke file & figure aslinya (`figures/rma_run13_curve.png`).
- **`delta_vs_best_f1_macro_pp` dan `is_tie_with_best` hanya valid per-`source`.** Keduanya dihitung
  `RunLogger.log` (`src/tuning.py:74`) relatif terhadap juara di dalam file asalnya. Jangan dipakai
  lintas-source, jangan dihitung ulang secara global.
- **Kolom efisiensi tidak boleh dibandingkan lintas-`source`.** Aturan validitas Bab 4: waktu latih,
  latensi, dan peak memory hanya sebanding bila diukur pada satu hardware + satu sesi. `source`
  berbeda = mesin/sesi berbeda. F1 tidak bergantung hardware, jadi aman dibandingkan.
- **Sel kosong = kolom itu memang tidak ada di file sumbernya**, bukan nilai hilang. Header antar
  campaign berbeda (mis. kolom `test_*` hanya ada di campaign yang membuka test set).
- `model_name` untuk `source == "vast"` diisi `indobenchmark/indobert-base-p2` oleh script
  (campaign itu berjalan sebelum kolom `model_name` ada; encoder-nya terdokumentasi di `CLAUDE.md`).

Angka final Bab 4 tetap dikutip dari `results/vast/`, bukan dari folder ini.

## Catatan data: `run_id` juara RM-c di `results/vast/best.json`

`best.json` mencatat juara RM-c sebagai `run_id: 13` dengan config `alpha=0.2, k=5, similarity`
dan `val_f1_macro=0.969903`. Di `runs_rmc.csv`, **config + F1 itu milik run #15**; run #13 justru
`k=1` dengan F1 0.969995. Jadi `run_id` di `best.json` bergeser, sementara **config dan F1-nya
benar** dan konsisten dengan yang dilaporkan di `CLAUDE.md`/Bab 4 (α=0.2, k=5, similarity).
Run #13 (`k=1`) memang sedikit lebih tinggi (+0,009 pp) tetapi sengaja tidak dipilih karena
`k=1` dinilai tidak stabil — lihat catatan di `run_candidate2_vast.sh`. Tidak diubah di sini:
`results/vast/` adalah angka Bab 4 yang sudah terkunci.
"""


def find_source_csv(dir_path: Path, scenario: str):
    """File ringkasan run untuk `scenario` di dalam `dir_path`.

    Menerima `runs_rma.csv` maupun sufiks hasil rename manual (`runs_rma_c2.csv`), tetapi
    MENOLAK `runs_rma_errors.csv` -- itu catatan config gagal (src/job_runner.py:365), bukan hasil run.
    """
    hits = [p for p in sorted(dir_path.glob(f"runs_{scenario}*.csv"))
            if not p.name.endswith("_errors.csv")]
    return hits


def order_columns(df: pd.DataFrame) -> list:
    lead = [c for c in LEAD_COLS if c in df.columns]
    return lead + [c for c in df.columns if c not in lead]


def merge_scenario(scenario: str):
    """Return (df_gabungan | None, laporan_baris_per_source)."""
    frames, report = [], []
    for label, d in SOURCES:
        if not d.exists():
            continue
        for p in find_source_csv(d, scenario):
            df = pd.read_csv(p)
            df.insert(0, "source", label)
            frames.append(df)
            report.append((label, p.relative_to(ROOT).as_posix(), len(df)))
    if not frames:
        return None, report
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "model_name" not in merged.columns:
        merged["model_name"] = pd.NA
    for label, name in MODEL_NAME_BACKFILL.items():
        fill = (merged["source"] == label) & merged["model_name"].isna()
        merged.loc[fill, "model_name"] = name
    return merged[order_columns(merged)], report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "combined"),
                    help="folder keluaran (default: results/combined)")
    ap.add_argument("--dry-run", action="store_true", help="cetak rencana saja, tidak menulis apa pun")
    args = ap.parse_args(argv)

    out = Path(args.out)
    print(f"Keluaran: {out}{'  [DRY RUN -- tidak menulis apa pun]' if args.dry_run else ''}\n")

    total_written = 0
    for sc in SCENARIOS:
        merged, report = merge_scenario(sc)
        if merged is None:
            print(f"[{sc}] tidak ada file sumber -- dilewati\n")
            continue
        print(f"[{sc}] runs_{sc}.csv  <- {len(report)} file, {len(merged)} baris")
        for label, rel, n in report:
            print(f"      {n:>4} baris  {rel}  (source={label})")
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            merged.to_csv(out / f"runs_{sc}.csv", index=False)
            total_written += 1
        print()

    if not args.dry_run:
        (out / "README.md").write_text(README, encoding="utf-8")
        print(f"Selesai: {total_written} file + README.md ditulis ke {out}")
        print("File sumber tidak disentuh sama sekali.")
    else:
        print("Dry run selesai -- tidak ada file yang ditulis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
