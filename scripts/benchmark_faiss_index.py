"""Jalankan & ukur modul pembangunan indeks FAISS (src/rac.py) -> catat ke CSV.

Yang diukur (memakai FUNGSI ASLI `rac.build_faiss_index` / `rac.retrieve`, bukan
reimplementasi, supaya angka merepresentasikan kode yang benar-benar dipakai RM-c):

1. Pembangunan indeks (`faiss_index_build.csv`)
   - waktu normalisasi L2, waktu `index.add`, total (diulang N kali -> mean/std/min)
   - ukuran indeks di memori (ntotal x dim x 4 byte), delta RSS proses
   - waktu serialisasi (`write_index`) + ukuran file, waktu muat ulang (`read_index`)
   - komposisi label vektor di dalam indeks (penting utk RAC: distribusi tetangga)
2. Latensi pencarian k-NN (`faiss_search_latency.csv`)
   - per (split query, k): total ms, ms/query, QPS

    python scripts/benchmark_faiss_index.py
    python scripts/benchmark_faiss_index.py --repeats 10 --k 1,3,5,10,20
    python scripts/benchmark_faiss_index.py --out-dir results/faiss_index_lite \
        --features results/vast/features/indobenchmark__indobert-base-p2

CATATAN VALIDITAS (aturan Bab 4): angka waktu/memori di sini terikat pada SATU
mesin + SATU sesi. Kolom `host`, `cpu`, `platform`, `faiss_threads` ikut dicatat
supaya ketahuan bila dicampur. Jangan gabungkan baris lintas-host ke satu tabel
efisiensi. Output DEFAULT sengaja BUKAN results/vast/ (folder itu memegang angka
Bab 4 yang sudah dikunci) melainkan results/faiss_index/.

Sifat script: aditif & idempoten-append -- file fitur (.npy) hanya dibaca, CSV
ditambahi baris baru (header ditulis sekali), tidak pernah menimpa baris lama.
"""

import argparse
import gc
import json
import os
import platform
import socket
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import rac  # noqa: E402  (butuh sys.path di atas)

DEFAULT_FEATURES = ROOT / "results" / "vast" / "features" / "indobenchmark__indobert-base-p2"
DEFAULT_OUT = ROOT / "results" / "faiss_index"


def rss_mb() -> float:
    """Resident set size proses (MB). FAISS mengalokasi di C++, jadi tracemalloc
    tidak melihatnya -- RSS proses adalah pengukur yang benar di sini."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return float("nan")


def host_info() -> dict:
    import faiss
    try:
        threads = faiss.omp_get_max_threads()
    except Exception:
        threads = -1
    return {
        "host": socket.gethostname(),
        "platform": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "faiss_threads": threads,
        "numpy_version": np.__version__,
    }


def append_csv(path: Path, rows: list[dict]) -> None:
    """Tambah baris ke CSV (buat + header bila belum ada). Tidak pernah menimpa."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        # kolom baru (versi script berbeda) tetap aman: union kolom, sisanya NaN
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def benchmark_build(train_emb, train_lab, repeats: int, info: dict, meta: dict,
                    out_dir: Path, save_index: bool):
    """Ulang pembangunan indeks `repeats` kali; kembalikan (rows, index_terakhir)."""
    import faiss

    norm_ms, add_ms, total_ms = [], [], []
    index = None
    for i in range(repeats):
        t0 = time.perf_counter()
        emb_n = rac.l2_normalize(train_emb)          # tahap 1: normalisasi L2 (cosine)
        t1 = time.perf_counter()
        index = faiss.IndexFlatIP(emb_n.shape[1])    # tahap 2: alokasi + add
        index.add(emb_n)
        t2 = time.perf_counter()
        norm_ms.append((t1 - t0) * 1000)
        add_ms.append((t2 - t1) * 1000)
        total_ms.append((t2 - t0) * 1000)
        print(f"  build #{i+1}/{repeats}: norm {norm_ms[-1]:.2f} ms | "
              f"add {add_ms[-1]:.2f} ms | total {total_ms[-1]:.2f} ms", flush=True)

    # Bangun sekali lagi lewat fungsi asli: (a) verifikasi bahwa yang diukur di atas
    # = jalur kode RM-c yang sesungguhnya, (b) ukur jejak RAM-nya dari kondisi bersih
    # (indeks loop dibuang dulu -- kalau tidak, delta RSS tertutup reuse alokator).
    del index
    gc.collect()
    rss_before = rss_mb()
    idx_ref, emb_ref = rac.build_faiss_index(train_emb)
    rss_after = rss_mb()   # mencakup indeks + salinan embedding ternormalisasi
    assert idx_ref.ntotal == len(train_emb) and idx_ref.d == train_emb.shape[1]
    index = idx_ref

    n, d = emb_ref.shape
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "module": "src.rac.build_faiss_index",
        "index_type": type(index).__name__,      # IndexFlatIP -> exact, bukan approx
        "metric": "inner_product (cosine, emb L2-normalized)",
        "model_name": meta.get("model_name"),
        "source_split": "train",                 # anti-leakage: index HANYA dari train
        "n_vectors": int(n),
        "dim": int(d),
        "dtype": str(emb_ref.dtype),
        "n_label_0": int((train_lab == 0).sum()),
        "n_label_1": int((train_lab == 1).sum()),
        "repeats": repeats,
        "norm_ms_mean": round(statistics.mean(norm_ms), 3),
        "add_ms_mean": round(statistics.mean(add_ms), 3),
        "build_total_ms_mean": round(statistics.mean(total_ms), 3),
        "build_total_ms_std": round(statistics.stdev(total_ms), 3) if repeats > 1 else 0.0,
        "build_total_ms_min": round(min(total_ms), 3),
        "build_total_ms_max": round(max(total_ms), 3),
        "index_mem_mb": round(n * d * 4 / (1024 ** 2), 3),   # IndexFlat = float32 mentah
        # delta RSS satu build bersih = indeks + salinan embedding ternormalisasi
        "rss_delta_mb": round(rss_after - rss_before, 2),
        "ntotal": int(index.ntotal),
        "is_trained": bool(index.is_trained),                # Flat: True tanpa training
    }

    if save_index:
        path = out_dir / "train_index.faiss"
        t0 = time.perf_counter()
        faiss.write_index(index, str(path))
        write_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        reloaded = faiss.read_index(str(path))
        read_ms = (time.perf_counter() - t0) * 1000
        assert reloaded.ntotal == index.ntotal
        row.update({
            "index_file_mb": round(path.stat().st_size / (1024 ** 2), 3),
            "write_index_ms": round(write_ms, 3),
            "read_index_ms": round(read_ms, 3),
            "index_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        })

    row.update(info)
    return [row], index


def benchmark_search(index, queries: dict, ks: list[int], repeats: int, info: dict,
                     meta: dict) -> list[dict]:
    rows = []
    for split, (emb, lab) in queries.items():
        for k in ks:
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                sims, idx = rac.retrieve(index, emb, k)
                times.append((time.perf_counter() - t0) * 1000)
            mean_ms = statistics.mean(times)
            # sanity: tetangga #1 utk query train adalah dirinya sendiri (sim ~1.0)
            rows.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "module": "src.rac.retrieve",
                "model_name": meta.get("model_name"),
                "query_split": split,
                "n_queries": int(emb.shape[0]),
                "k": k,
                "repeats": repeats,
                "search_total_ms_mean": round(mean_ms, 3),
                "search_total_ms_std": round(statistics.stdev(times), 3) if repeats > 1 else 0.0,
                "ms_per_query": round(mean_ms / emb.shape[0], 5),
                "queries_per_sec": round(emb.shape[0] / (mean_ms / 1000), 1),
                "mean_top1_sim": round(float(sims[:, 0].mean()), 4),
                "mean_topk_sim": round(float(sims.mean()), 4),
                **info,
            })
            print(f"  search {split:<5} k={k:<3}: {mean_ms:8.2f} ms total | "
                  f"{mean_ms / emb.shape[0]:.4f} ms/query", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                    help="folder embedding beku (*_emb.npy / *_label.npy)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="folder output CSV (JANGAN results/vast: angka Bab 4 terkunci)")
    ap.add_argument("--repeats", type=int, default=5, help="pengulangan tiap pengukuran")
    ap.add_argument("--k", default="1,3,5,10,20", help="daftar k untuk uji pencarian")
    ap.add_argument("--no-search", action="store_true", help="lewati benchmark pencarian")
    ap.add_argument("--no-save-index", action="store_true",
                    help="jangan tulis train_index.faiss (lewati ukur write/read)")
    args = ap.parse_args()

    feat = args.features
    if not (feat / "train_emb.npy").exists():
        raise SystemExit(f"embedding train tidak ditemukan di {feat}. "
                         "Jalankan ekstraksi fitur dulu (app.py / job_runner).")

    meta_path = feat / "extract_meta.json"
    meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else {}
    info = host_info()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_emb = np.load(feat / "train_emb.npy")
    train_lab = np.load(feat / "train_label.npy")
    print(f"[faiss] fitur: {feat.relative_to(ROOT)}")
    print(f"[faiss] encoder: {meta.get('model_name', '?')} | train_emb {train_emb.shape}")
    print(f"[faiss] host: {info['host']} | {info['cpu']} | faiss {info['faiss_version']} "
          f"({info['faiss_threads']} thread)")

    print("\n== Pembangunan indeks ==")
    build_rows, index = benchmark_build(train_emb, train_lab, args.repeats, info, meta,
                                        out_dir, save_index=not args.no_save_index)
    build_csv = out_dir / "faiss_index_build.csv"
    append_csv(build_csv, build_rows)
    print(f"-> {build_csv.relative_to(ROOT)}")

    if not args.no_search:
        print("\n== Latensi pencarian k-NN ==")
        queries = {s: (np.load(feat / f"{s}_emb.npy"), np.load(feat / f"{s}_label.npy"))
                   for s in ["val", "test"] if (feat / f"{s}_emb.npy").exists()}
        ks = [int(x) for x in args.k.split(",") if x.strip()]
        search_rows = benchmark_search(index, queries, ks, args.repeats, info, meta)
        search_csv = out_dir / "faiss_search_latency.csv"
        append_csv(search_csv, search_rows)
        print(f"-> {search_csv.relative_to(ROOT)}")

    r = build_rows[0]
    print(f"\nRingkas: {r['n_vectors']} vektor x {r['dim']} dim -> {r['index_type']} "
          f"dibangun {r['build_total_ms_mean']:.2f} ms (±{r['build_total_ms_std']:.2f}), "
          f"{r['index_mem_mb']:.2f} MB di memori.")


if __name__ == "__main__":
    main()
