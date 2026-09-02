"""Gabungkan SEMUA berkas nilai (CSV + JSON) hasil penelitian menjadi SATU workbook: HASIL.xlsx.

Masalah yang dipecahkan: nilai hasil penelitian tersebar di puluhan CSV/JSON dalam banyak folder
(`results/vast/`, `results/faiss_index/`, `results/vast_rmc_cheap_head/`, `results_c2/`, ...),
sehingga sulit tahu angka mana yang dipakai. Script ini menyalin seluruhnya ke satu berkas Excel
ber-sheet, plus tiga sheet gabungan (`02_SEMUA_NILAI`, `03_SEMUA_RUN`, `04_KURVA_EPOCH`) yang
menyatukan baris dari semua campaign dengan kolom `sumber` sebagai penanda asal.

    python scripts/build_hasil_workbook.py
    python scripts/build_hasil_workbook.py --out HASIL.xlsx

Sifat: READ-ONLY terhadap seluruh berkas hasil -- hanya membaca lalu menulis satu berkas keluaran.
Menjalankan ulang cukup menimpa keluaran itu; tidak ada artefak sumber yang tersentuh.

CATATAN CAKUPAN:
- `dataset/splits/*.csv` (9.395 baris teks komentar) SENGAJA tidak dimasukkan -- itu data masukan,
  bukan nilai hasil, dan memuat teks personal. Ringkasannya ada di sheet `11_DATASET`.
- Kolom efisiensi (waktu/memori/latency) TIDAK boleh dibandingkan lintas nilai `sumber`
  (aturan validitas Bab 4: satu hardware + satu sesi). F1 aman dibandingkan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# (label sumber, folder) -- label dipakai sebagai nilai kolom `sumber` di sheet gabungan.
VAST = ROOT / "results" / "vast"
CHEAP = ROOT / "results" / "vast_rmc_cheap_head"
C2 = ROOT / "results_c2" / "vast_candidate2"
FAISS = ROOT / "results" / "faiss_index"

STATUS = {
    "vast": "RESMI (angka Bab 4)",
    "cheap_head": "pendukung (ablasi head murah, mesin/sesi lain)",
    "candidate2": "pendukung (kandidat #2, sesi lain)",
    "faiss_bench": "pendukung (laptop CPU, bukan RTX 3090)",
    "dataset": "RESMI (data siap latih)",
}


# --------------------------------------------------------------------------- util

def read_csv(path: Path) -> pd.DataFrame | None:
    """Baca CSV bila ada; None bila tidak (folder eksperimen bersifat opsional)."""
    if not path.exists():
        print(f"  [lewat] tidak ada: {path.relative_to(ROOT)}")
        return None
    return pd.read_csv(path)


def read_json(path: Path):
    if not path.exists():
        print(f"  [lewat] tidak ada: {path.relative_to(ROOT)}")
        return None
    return json.load(open(path, encoding="utf-8"))


def flatten(obj, prefix: str = "") -> list[tuple[str, object]]:
    """JSON bersarang -> daftar (kunci_bertitik, nilai skalar)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += flatten(v, f"{prefix}[{i}]")
    else:
        out.append((prefix, obj))
    return out


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# --------------------------------------------------------------------------- sheet

def sheet_semua_run() -> pd.DataFrame:
    """Semua baris run dari semua campaign, kolom disatukan (union)."""
    spec = [
        ("vast", "rma", VAST / "runs_rma.csv"),
        ("vast", "rmb", VAST / "runs_rmb.csv"),
        ("vast", "rmc", VAST / "runs_rmc.csv"),
        ("cheap_head", "rmb", CHEAP / "runs_rmb_cheap.csv"),
        ("cheap_head", "rmc", CHEAP / "runs_rmc_cheap.csv"),
        ("candidate2", "rma", C2 / "runs_rma_c2.csv"),
        ("candidate2", "rmb", C2 / "runs_rmb_c2.csv"),
        ("candidate2", "rmc", C2 / "runs_rmc_c2.csv"),
    ]
    frames = []
    for sumber, skenario, path in spec:
        d = read_csv(path)
        if d is None:
            continue
        d = d.copy()
        d.insert(0, "sumber", sumber)
        d.insert(1, "status_sumber", STATUS[sumber])
        if "scenario" not in d.columns:
            d.insert(2, "scenario", skenario)
        d.insert(3, "sumber_file", rel(path))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)
    # urutan kolom: penanda -> identitas run -> sisanya apa adanya
    lead = ["sumber", "status_sumber", "scenario", "sumber_file", "run_id", "batch_id", "timestamp"]
    lead = [c for c in lead if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]


def sheet_kurva_epoch() -> pd.DataFrame:
    """Kurva validasi per epoch dari semua campaign."""
    frames = []
    for sumber, skenario, path in [
        ("vast", "rma", VAST / "history" / "rma_history.csv"),
        ("vast", "rmb", VAST / "history" / "rmb_history.csv"),
        ("cheap_head", "rmb", CHEAP / "history" / "rmb_history.csv"),
    ]:
        d = read_csv(path)
        if d is None:
            continue
        d = d.copy()
        d.insert(0, "sumber", sumber); d.insert(1, "scenario", skenario)
        d.insert(2, "sumber_file", rel(path))
        frames.append(d)
    # candidate2 memakai format lama: satu file per run, tanpa kolom run_id
    for path in sorted((C2 / "history").glob("*.csv")) if (C2 / "history").exists() else []:
        d = pd.read_csv(path)
        nama = path.stem                      # mis. rma_run1_c2
        skenario = nama.split("_")[0]
        run_id = "".join(ch for ch in nama.split("_")[1] if ch.isdigit())
        d.insert(0, "sumber", "candidate2"); d.insert(1, "scenario", skenario)
        d.insert(2, "sumber_file", rel(path))
        d.insert(3, "run_id", int(run_id) if run_id else None)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)
    lead = ["sumber", "scenario", "sumber_file", "run_id", "epoch"]
    return df[[c for c in lead if c in df.columns] + [c for c in df.columns if c not in lead]]


def sheet_grid_pivot() -> pd.DataFrame:
    """Semua pivot grid (matriks) diubah ke bentuk panjang agar muat dalam satu sheet."""
    rows = []
    spec = [("vast", VAST / "metrics"), ("cheap_head", CHEAP / "metrics")]
    for sumber, mdir in spec:
        if not mdir.exists():
            continue
        for path in sorted(mdir.glob("*_grid_pivot_*.csv")):
            skenario = path.stem.split("_")[0]                 # rma / rmb / rmc
            kondisi = path.stem.split("_grid_pivot_")[1]       # batch16, hidden_dim1024, ...
            d = pd.read_csv(path)
            sumbu_baris = d.columns[0]                          # lr atau alpha
            sumbu_kolom = "k" if skenario == "rmc" else "epochs"
            long = d.melt(id_vars=[sumbu_baris], var_name=sumbu_kolom, value_name="val_f1_macro")
            for _, r in long.iterrows():
                rows.append({
                    "sumber": sumber, "scenario": skenario, "kondisi": kondisi,
                    "sumbu_baris": sumbu_baris, "nilai_baris": r[sumbu_baris],
                    "sumbu_kolom": sumbu_kolom, "nilai_kolom": r[sumbu_kolom],
                    "val_f1_macro": r["val_f1_macro"], "sumber_file": rel(path),
                })
    return pd.DataFrame(rows)


def sheet_metrik_final() -> pd.DataFrame:
    """final_comparison + inference_benchmark + success_criteria -> satu tabel panjang."""
    rows = []
    for path, kelompok in [
        (VAST / "metrics" / "final_comparison.csv", "perbandingan final (test)"),
        (VAST / "metrics" / "inference_benchmark.csv", "benchmark inferensi"),
        (VAST / "metrics" / "success_criteria.csv", "kriteria sukses"),
    ]:
        d = read_csv(path)
        if d is None:
            continue
        kunci = "model" if "model" in d.columns else "scenario"
        for _, r in d.iterrows():
            for col in d.columns:
                if col == kunci:
                    continue
                rows.append({"kelompok": kelompok, "model": r[kunci], "metrik": col,
                             "nilai": r[col], "sumber_file": rel(path)})
    return pd.DataFrame(rows)


def sheet_semua_nilai() -> pd.DataFrame:
    """Setiap nilai skalar dari seluruh berkas JSON + metrik final, satu baris satu nilai."""
    rows = []

    def add(sumber, kelompok, path: Path, obj, satuan=""):
        if obj is None:
            return
        for kunci, nilai in flatten(obj):
            rows.append({"sumber": sumber, "status_sumber": STATUS.get(sumber, ""),
                         "kelompok": kelompok, "kunci": kunci, "nilai": nilai,
                         "satuan": satuan, "sumber_file": rel(path)})

    p = ROOT / "dataset" / "processed" / "metadata.json"
    add("dataset", "dataset & preprocessing", p, read_json(p))
    for sumber, base in [("vast", VAST), ("cheap_head", CHEAP)]:
        add(sumber, "config juara per skenario", base / "best.json", read_json(base / "best.json"))
        add(sumber, "ringkasan campaign", base / "tuning_summary.json",
            read_json(base / "tuning_summary.json"))
        add(sumber, "hardware", base / "hardware.json", read_json(base / "hardware.json"))
        fdir = base / "features" / "indobenchmark__indobert-base-p2" / "extract_meta.json"
        add(sumber, "ekstraksi fitur beku", fdir, read_json(fdir))
    add("candidate2", "config juara per skenario", C2 / "best_c2.json", read_json(C2 / "best_c2.json"))
    add("candidate2", "hardware", C2 / "hardware_c2.json", read_json(C2 / "hardware_c2.json"))
    add("candidate2", "ringkasan campaign", C2 / "tuning_summary_c2.json",
        read_json(C2 / "tuning_summary_c2.json"))
    c2_extract = C2 / "features" / "indobenchmark__indobert-base-p2" / "extract_meta_c2.json"
    add("candidate2", "ekstraksi fitur beku", c2_extract, read_json(c2_extract))

    # metrik final ikut masuk agar sheet ini benar-benar memuat SEMUA nilai
    mf = sheet_metrik_final()
    for _, r in mf.iterrows():
        rows.append({"sumber": "vast", "status_sumber": STATUS["vast"], "kelompok": r["kelompok"],
                     "kunci": f"{r['model']}.{r['metrik']}", "nilai": r["nilai"], "satuan": "",
                     "sumber_file": r["sumber_file"]})
    return pd.DataFrame(rows)


def sheet_ringkasan() -> pd.DataFrame:
    """Tabel utama Bab 4: performa + efisiensi + verdict kriteria, satu baris per model."""
    comp = read_csv(VAST / "metrics" / "final_comparison.csv")
    crit = read_csv(VAST / "metrics" / "success_criteria.csv")
    if comp is None:
        return pd.DataFrame()
    df = comp.copy()
    if crit is not None:
        df = df.merge(crit.drop(columns=[c for c in ["train_time_s", "rma_train_time_s"]
                                         if c in crit.columns]), on="model", how="left")
    # peak memory latih diambil dari best.json (tidak ada di final_comparison)
    best = read_json(VAST / "best.json") or {}
    peta = {"RM-a": "rma", "RM-b": "rmb", "RM-c": "rmc"}
    df["train_peak_mem_mib"] = df["model"].map(
        lambda m: (best.get(peta.get(m, ""), {}) or {}).get("peak_mem_mb"))
    df["train_peak_mem_mb_desimal"] = df["train_peak_mem_mib"].map(
        lambda v: round(v * 1.048576, 1) if pd.notna(v) else None)
    # RM-a adalah pembanding kriteria sukses (bukan yang dinilai); RM-c tidak melatih apa pun.
    for kol in ["f1_gap_pp", "lolos_f1_gap<=3pp", "param_reduction_pct", "lolos_param>=90%",
                "time_reduction_pct", "lolos_waktu>=50%", "kriteria_terpenuhi", "kompetitif(>=2/3)"]:
        if kol in df.columns:
            df.loc[df["model"] == "RM-a", kol] = "(baseline pembanding)"
    for kol in ["train_peak_mem_mib", "train_peak_mem_mb_desimal"]:
        df.loc[df["model"] == "RM-c", kol] = "(tanpa training)"
    df["catatan"] = df["model"].map({
        "RM-a": "baseline full fine-tuning",
        "RM-b": "encoder beku + MLP head (1024); train_time = 7,90 s ekstraksi + 3,75 s head",
        "RM-c": "RAC di atas head RM-b; tanpa training; config = alpha 0,2 / k 5 / similarity "
                "(baris run #15 di runs_rmc.csv -- best.json menulis run_id 13, itu keliru)",
    })
    return df


def sheet_index(sheets: dict[str, pd.DataFrame], deskripsi: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{"sheet": nama, "isi": deskripsi.get(nama, ""),
                          "jumlah_baris": len(df), "jumlah_kolom": len(df.columns)}
                         for nama, df in sheets.items() if nama != "00_INDEX"])


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "HASIL.xlsx")
    args = ap.parse_args()

    print("Membaca berkas nilai...")
    sheets: dict[str, pd.DataFrame] = {}

    sheets["01_RINGKASAN_BAB4"] = sheet_ringkasan()
    sheets["02_SEMUA_NILAI"] = sheet_semua_nilai()
    sheets["03_SEMUA_RUN"] = sheet_semua_run()
    sheets["04_KURVA_EPOCH"] = sheet_kurva_epoch()
    sheets["05_GRID_PIVOT"] = sheet_grid_pivot()
    sheets["06_METRIK_FINAL"] = sheet_metrik_final()

    for nama, path in [
        ("07_FAISS_BUILD", FAISS / "faiss_index_build.csv"),
        ("08_FAISS_SEARCH", FAISS / "faiss_search_latency.csv"),
        ("09_LATENCY_GPU_LAIN", CHEAP / "metrics_latency_local_rtx3050.csv"),
        ("10_KANDIDAT2", ROOT / "results_c2" / "final_comparison_all_candidates.csv"),
        ("13_BAB4_DATA_MENTAH", ROOT / "draft_bab4" / "data_mentah_bab4.csv"),
    ]:
        d = read_csv(path)
        if d is not None:
            d = d.copy(); d["sumber_file"] = rel(path)
            sheets[nama] = d

    meta = read_json(ROOT / "dataset" / "processed" / "metadata.json") or {}
    sheets["11_DATASET"] = pd.DataFrame(
        [{"kelompok": k.split(".")[0], "kunci": k, "nilai": v,
          "sumber_file": "dataset/processed/metadata.json"} for k, v in flatten(meta)])

    hw_rows = []
    for sumber, path in [("vast", VAST / "hardware.json"), ("cheap_head", CHEAP / "hardware.json"),
                         ("candidate2", C2 / "hardware_c2.json")]:
        obj = read_json(path)
        if obj:
            hw_rows.append({"sumber": sumber, "status_sumber": STATUS.get(sumber, ""),
                            **obj, "sumber_file": rel(path)})
    sheets["12_HARDWARE"] = pd.DataFrame(hw_rows)

    # Berkas nilai yang sengaja TIDAK disalin -- didaftarkan supaya tidak ada yang hilang diam-diam.
    lewat = [
        ("results/combined/runs_{rma,rmb,rmc}.csv", 3,
         "turunan scripts/merge_runs.py -- barisnya sudah ada di sheet 03_SEMUA_RUN"),
        ("results/vast/history/_backup_per_run/*.csv", 57,
         "format lama satu-file-per-run; sudah digabung ke 04_KURVA_EPOCH"),
        ("results/vast_rmc_cheap_head/history/_backup_per_run/*.csv", 1,
         "sama seperti di atas"),
        ("results/**/progress.json, results/vast/job.json", 4,
         "status job (running/done/pesan), bukan nilai hasil"),
        ("dataset/splits/{train,val,test}.csv", 3,
         "data MASUKAN (9.395 baris teks komentar, memuat teks personal); ringkasannya di 11_DATASET"),
        ("dataset/raw/data_labeling.csv", 1,
         "data mentah sebelum preprocessing; ringkasannya di 11_DATASET"),
        ("tuning_grids/*.csv", 8,
         "rancangan grid (rencana input), bukan nilai hasil; hasilnya ada di 03_SEMUA_RUN"),
    ]
    sheets["14_TIDAK_DIMASUKKAN"] = pd.DataFrame(
        [{"pola_berkas": p, "jumlah_berkas": n, "alasan": a} for p, n, a in lewat])

    deskripsi = {
        "01_RINGKASAN_BAB4": "TABEL UTAMA: performa test + efisiensi + verdict kriteria sukses (3 model)",
        "02_SEMUA_NILAI": "Setiap nilai skalar dari SEMUA berkas JSON + metrik final -- satu baris satu nilai",
        "03_SEMUA_RUN": "Semua run tuning dari semua campaign (kolom `sumber` = asal folder)",
        "04_KURVA_EPOCH": "Metrik validasi per epoch tiap run (untuk deteksi overfitting)",
        "05_GRID_PIVOT": "Semua matriks grid (lr x epochs, alpha x k) dalam bentuk panjang",
        "06_METRIK_FINAL": "final_comparison + inference_benchmark + success_criteria (bentuk panjang)",
        "07_FAISS_BUILD": "Benchmark pembangunan indeks FAISS -- LAPTOP CPU, bukan RTX 3090",
        "08_FAISS_SEARCH": "Benchmark pencarian k-NN -- LAPTOP CPU, bukan RTX 3090",
        "09_LATENCY_GPU_LAIN": "Latency inferensi di RTX 3050 -- jangan dicampur dengan angka 3090",
        "10_KANDIDAT2": "Perbandingan config resmi vs runner-up (uji sensitivitas)",
        "11_DATASET": "metadata.json: jumlah baris tiap tahap, komposisi split, class weight",
        "12_HARDWARE": "Spesifikasi mesin tiap campaign",
        "13_BAB4_DATA_MENTAH": "Tabel nilai per-subbab yang sudah disiapkan untuk draf Bab 4",
        "14_TIDAK_DIMASUKKAN": "Berkas yang sengaja tidak disalin ke sini + alasannya",
    }
    sheets = dict(sorted(sheets.items()))                       # urut 01..14
    sheets = {"00_INDEX": sheet_index(sheets, deskripsi), **sheets}

    print(f"\nMenulis {args.out.name} ...")
    with pd.ExcelWriter(args.out, engine="openpyxl") as xl:
        for nama, df in sheets.items():
            df.to_excel(xl, sheet_name=nama[:31], index=False)
            ws = xl.sheets[nama[:31]]
            ws.freeze_panes = "A2"
            if len(df):
                ws.auto_filter.ref = ws.dimensions
            for i, col in enumerate(df.columns, start=1):
                lebar = len(str(col))
                for v in df[col].head(200):
                    lebar = max(lebar, min(len(str(v)), 60))
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(max(lebar + 2, 10), 62)
            print(f"  {nama:22s} {len(df):5d} baris x {len(df.columns):2d} kolom")

    print(f"\nSelesai -> {rel(args.out)}")


if __name__ == "__main__":
    main()
