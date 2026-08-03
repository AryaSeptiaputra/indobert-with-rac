"""Worker training/evaluasi SATU KONFIGURASI — dijalankan sebagai proses terpisah oleh app.py.

Alur kerja: 1 klik = 1 config = 1 run. Tidak ada pencarian otomatis; keputusan nilai
hyperparameter berikutnya diambil manusia (dibantu analisis) setelah melihat hasil run.
Semua run MENUMPUK di `<out>/runs_{rma,rmb,rmc}.csv` beserta kolom `catatan` (alasan).

Kenapa proses terpisah: Streamlit me-rerun skrip tiap interaksi → job panjang tak boleh
hidup di dalam callback UI. Dengan subprocess, job tetap jalan meski tab ditutup.

Pakai:
    python src/job_runner.py --config path/ke/job.json

job.json:
{
  "phase": "rma" | "rmb" | "rmc" | "final",
  "out_dir": "results/vast",
  "smoke": false,
  "config": { ...nilai hyperparameter... },
  "note": "alasan memilih nilai ini",
  "eval_test": false
}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import evaluate as E
import modeling as M
import reporting
import tuning as T
from dataset import GamblingCommentDataset, load_tokenizer

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------- util -----------------------------

def write_progress(out: Path, **kw):
    p = Path(out) / "progress.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = {}
    if p.exists():
        try:
            cur = json.load(open(p, encoding="utf-8"))
        except Exception:
            cur = {}
    cur.update(kw)
    cur["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(cur, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def write_hardware(out: Path):
    info = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "torch": torch.__version__,
            "vram_total_mb": (round(torch.cuda.get_device_properties(0).total_memory / 1024**2)
                              if torch.cuda.is_available() else 0)}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        info["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        pass
    json.dump(info, open(Path(out) / "hardware.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return info


def read_json(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def update_best(out: Path, scenario: str, payload: dict):
    """Simpan info config TERBAIK-sejauh-ini (by val F1-macro) per skenario."""
    p = out / "best.json"
    data = read_json(p)
    prev = data.get(scenario, {})
    if payload["val_f1_macro"] > prev.get("val_f1_macro", -1):
        data[scenario] = payload
        json.dump(data, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  -> BEST BARU untuk {scenario} (val F1 {payload['val_f1_macro']:.4f})", flush=True)
        return True
    return False


def build_ctx(cfg, device):
    data_dir = ROOT / "dataset" / "splits"
    meta = ROOT / "dataset" / "processed" / "metadata.json"
    dfs = {s: pd.read_csv(data_dir / f"{s}.csv") for s in ["train", "val", "test"]}
    if cfg.get("smoke"):
        dfs = {s: d.sample(min(len(d), {"train": 200, "val": 100, "test": 100}[s]),
                           random_state=42).reset_index(drop=True) for s, d in dfs.items()}
        print(f"[smoke] subset: {[len(dfs[s]) for s in dfs]}", flush=True)
    cw = json.load(open(meta, encoding="utf-8"))["class_weights"]
    weight = torch.tensor([cw["0"], cw["1"]], dtype=torch.float, device=device)
    tokenizer = load_tokenizer(cfg.get("model_name", "indobenchmark/indobert-base-p2"))
    return {"train_df": dfs["train"], "val_df": dfs["val"], "test_df": dfs["test"],
            "tokenizer": tokenizer, "weight": weight, "device": device,
            "model_name": cfg.get("model_name", "indobenchmark/indobert-base-p2"),
            "max_length": cfg.get("max_length", 128), "num_workers": 0}


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def features_dir(ctx, out: Path) -> Path:
    return out / "features" / model_slug(ctx["model_name"])


def ensure_features(ctx, out: Path):
    fdir = features_dir(ctx, out); fdir.mkdir(parents=True, exist_ok=True)
    if not all((fdir / f"{s}_emb.npy").exists() for s in ["train", "val", "test"]):
        print(f"[features] ekstraksi fitur beku (encoder {ctx['model_name']}, sekali saja)...", flush=True)
        from torch.utils.data import DataLoader
        enc = M.build_encoder(ctx["tokenizer"], model_name=ctx["model_name"]).to(ctx["device"])
        t0 = time.perf_counter()
        if ctx["device"].type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        hidden_dim = None
        for s, df in [("train", ctx["train_df"]), ("val", ctx["val_df"]), ("test", ctx["test_df"])]:
            ds = GamblingCommentDataset(df["text_clean"], df["label"], tokenizer=ctx["tokenizer"],
                                        max_length=ctx["max_length"])
            emb, lab = M.extract_features(enc, DataLoader(ds, batch_size=32), ctx["device"], use_amp=True)
            np.save(fdir / f"{s}_emb.npy", emb); np.save(fdir / f"{s}_label.npy", lab)
            hidden_dim = emb.shape[1]
            print(f"[features] {s}: {emb.shape}", flush=True)
        json.dump({"model_name": ctx["model_name"], "hidden_dim": hidden_dim,
                   "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "extract_time_s": round(time.perf_counter() - t0, 1),
                   "extract_peak_gpu_mem_mb": round(E.peak_gpu_mem_mb(), 1)},
                  open(fdir / "extract_meta.json", "w"), indent=2)
        del enc
        if ctx["device"].type == "cuda":
            torch.cuda.empty_cache()
    return {s: (np.load(fdir / f"{s}_emb.npy"), np.load(fdir / f"{s}_label.npy"))
            for s in ["train", "val", "test"]}


def load_best_head(out: Path, device):
    ck = torch.load(out / "checkpoints" / "rmb_best.pt", map_location=device)
    c = ck["config"]
    head = M.build_head(c["head_arch"], hidden_size=ck["hidden_size"], dropout=c["dropout"],
                        hidden_dim=c.get("hidden_dim", 256))
    head.load_state_dict(ck["head_state"])
    # Checkpoint lama (pra fitur ini) tidak punya key ini -- None berarti "tak diketahui",
    # bukan berarti pasti cocok.
    c["_encoder_model_name"] = ck.get("model_name")
    return head, c


# ----------------------------- fase (1 run = 1 config) -----------------------------

def _run_one_rma(cfg_input, note, eval_test, ctx, out, logger, batch_id=""):
    """Latih+evaluasi RM-a untuk SATU config; log 1 baris; return row (sudah di-log).

    Dipakai baik oleh phase_rma (1 config) maupun run_batch (banyak config).
    """
    cfg = {**T.RMA_DEFAULT, **cfg_input}
    rid = logger.next_id()
    print(f"[rma] RUN #{rid} config={cfg}", flush=True)
    vm, hist, extra = T.train_eval_rma(cfg, ctx)
    row = {"scenario": "rma", "batch_id": batch_id, "model_name": ctx["model_name"], **cfg,
           "micro_batch_eff": extra["micro_batch"],
           "grad_accum": extra["grad_accum"],
           "val_f1_macro": round(vm["f1_macro"], 6), "val_acc": round(vm["accuracy"], 6),
           "val_precision_macro": round(vm["precision_macro"], 6),
           "val_recall_macro": round(vm["recall_macro"], 6),
           "val_f1_judi": round(vm["f1_class1"], 6),
           "best_epoch": extra["best_epoch"], "train_time_s": extra["train_time_s"],
           "peak_mem_mb": extra["peak_mem_mb"], "trainable_params": extra["trainable_params"],
           "catatan": note}
    if eval_test:
        model = extra["model"]; model.load_state_dict(extra["best_state"])
        tm, yt, pt, prt = T.test_rma(model, ctx, cfg.get("micro_batch", 32))
        row.update({f"test_{k}": round(v, 6) for k, v in tm.items()
                    if k in ["f1_macro", "accuracy", "f1_class1", "precision_class1", "recall_class1"]})
        E.plot_confusion_matrix(yt, pt, out / "figures" / f"rma_run{rid}_confusion.png",
                                title=f"RM-a run#{rid} (test)")
        E.plot_pr_curve(yt, prt, out / "figures" / f"rma_run{rid}_pr.png",
                        title=f"RM-a run#{rid} (test)")
        print(f"[rma] TEST F1-macro {tm['f1_macro']:.4f}", flush=True)
    row = logger.log(row)
    pd.DataFrame(hist).to_csv(out / "history" / f"rma_run{rid}.csv", index=False)
    try:
        reporting.training_curve(out, "rma", rid, hist)
    except Exception as e:
        print(f"[rma] gagal membuat kurva training (diabaikan): {e}", flush=True)
    if update_best(out, "rma", {"run_id": rid, "config": cfg, "val_f1_macro": vm["f1_macro"],
                                "train_time_s": extra["train_time_s"],
                                "trainable_params": extra["trainable_params"],
                                "peak_mem_mb": extra["peak_mem_mb"]}):
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": extra["best_state"], "config": cfg, "run_id": rid,
                    "val_f1_macro": vm["f1_macro"]}, out / "checkpoints" / "rma_best.pt")
        try:
            yv, pv, prv = T._predict(extra["model"],
                                     T._loader(ctx["val_df"], cfg.get("micro_batch", 32), False, ctx), ctx["device"])
            E.plot_confusion_matrix(yv, pv, out / "figures" / "rma_best_val_confusion.png",
                                    title=f"RM-a juara saat ini (run#{rid}, val)")
            E.plot_pr_curve(yv, prv, out / "figures" / "rma_best_val_pr.png",
                            title=f"RM-a juara saat ini (run#{rid}, val)")
        except Exception as e:
            print(f"[rma] gagal membuat confusion matrix/PR curve juara (diabaikan): {e}", flush=True)
    print(f"[rma] RUN #{rid} val F1-macro {vm['f1_macro']:.4f} (best epoch {extra['best_epoch']})", flush=True)
    return row


def phase_rma(job, ctx, out):
    logger = T.RunLogger(out / "runs_rma.csv")
    write_progress(out, phase="rma", status="running", message=f"RM-a run #{logger.next_id()}")
    _run_one_rma(job.get("config", {}), job.get("note", ""), bool(job.get("eval_test")), ctx, out, logger)


def _run_one_rmb(cfg_input, note, eval_test, ctx, out, logger, batch_id="", feats=None):
    """Latih+evaluasi RM-b untuk SATU config; log 1 baris; return row (sudah di-log)."""
    cfg = {**T.RMB_DEFAULT, **cfg_input}
    rid = logger.next_id()
    print(f"[rmb] RUN #{rid} config={cfg}", flush=True)
    vm, hist, extra = T.train_eval_rmb(cfg, feats, ctx["weight"], ctx["device"], return_test=eval_test)
    ex_meta = read_json(features_dir(ctx, out) / "extract_meta.json")
    row = {"scenario": "rmb", "batch_id": batch_id, "model_name": ctx["model_name"], **cfg,
           "val_f1_macro": round(vm["f1_macro"], 6), "val_acc": round(vm["accuracy"], 6),
           "val_precision_macro": round(vm["precision_macro"], 6),
           "val_recall_macro": round(vm["recall_macro"], 6),
           "val_f1_judi": round(vm["f1_class1"], 6),
           "best_epoch": extra["best_epoch"], "head_train_time_s": extra["train_time_s"],
           "extract_time_s": ex_meta.get("extract_time_s"),
           "train_time_s": round(extra["train_time_s"] + (ex_meta.get("extract_time_s") or 0), 2),
           "trainable_params": extra["trainable_params"], "peak_mem_mb": extra["peak_mem_mb"],
           "infer_latency_ms": extra["infer_latency_ms"], "catatan": note}
    if eval_test:
        tm = extra["test_metrics"]
        row.update({f"test_{k}": round(v, 6) for k, v in tm.items()
                    if k in ["f1_macro", "accuracy", "f1_class1", "precision_class1", "recall_class1"]})
        E.plot_confusion_matrix(feats["test"][1], extra["test_pred"],
                                out / "figures" / f"rmb_run{rid}_confusion.png",
                                title=f"RM-b run#{rid} (test)")
        E.plot_pr_curve(feats["test"][1], extra["test_pred_proba"],
                        out / "figures" / f"rmb_run{rid}_pr.png", title=f"RM-b run#{rid} (test)")
        print(f"[rmb] TEST F1-macro {tm['f1_macro']:.4f}", flush=True)
    row = logger.log(row)
    pd.DataFrame(hist).to_csv(out / "history" / f"rmb_run{rid}.csv", index=False)
    try:
        reporting.training_curve(out, "rmb", rid, hist)
    except Exception as e:
        print(f"[rmb] gagal membuat kurva training (diabaikan): {e}", flush=True)
    if update_best(out, "rmb", {"run_id": rid, "config": cfg, "val_f1_macro": vm["f1_macro"],
                                "train_time_s": row["train_time_s"],
                                "trainable_params": extra["trainable_params"],
                                "peak_mem_mb": extra["peak_mem_mb"],
                                "infer_latency_ms": extra["infer_latency_ms"],
                                "model_name": ctx["model_name"]}):
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save({"head_state": extra["best_state"], "config": cfg, "run_id": rid,
                    "hidden_size": extra["hidden_size"], "val_f1_macro": vm["f1_macro"],
                    "model_name": ctx["model_name"]},
                   out / "checkpoints" / "rmb_best.pt")
        try:
            head = M.build_head(cfg["head_arch"], hidden_size=extra["hidden_size"], dropout=cfg["dropout"],
                                hidden_dim=cfg.get("hidden_dim", 256))
            head.load_state_dict(extra["best_state"]); head.to(ctx["device"]).eval()
            with torch.no_grad():
                logits_v = head(torch.tensor(feats["val"][0], device=ctx["device"]))
                pv = logits_v.argmax(1).cpu().numpy()
                prv = torch.softmax(logits_v.float(), dim=-1)[:, 1].cpu().numpy()
            E.plot_confusion_matrix(feats["val"][1], pv, out / "figures" / "rmb_best_val_confusion.png",
                                    title=f"RM-b juara saat ini (run#{rid}, val)")
            E.plot_pr_curve(feats["val"][1], prv, out / "figures" / "rmb_best_val_pr.png",
                            title=f"RM-b juara saat ini (run#{rid}, val)")
        except Exception as e:
            print(f"[rmb] gagal membuat confusion matrix/PR curve juara (diabaikan): {e}", flush=True)
    print(f"[rmb] RUN #{rid} val F1-macro {vm['f1_macro']:.4f} (best epoch {extra['best_epoch']})", flush=True)
    return row


def phase_rmb(job, ctx, out):
    feats = ensure_features(ctx, out)
    logger = T.RunLogger(out / "runs_rmb.csv")
    write_progress(out, phase="rmb", status="running", message=f"RM-b run #{logger.next_id()}")
    _run_one_rmb(job.get("config", {}), job.get("note", ""), bool(job.get("eval_test")), ctx, out, logger,
                feats=feats)


def _run_one_rmc(cfg_input, note, eval_test, ctx, out, logger, batch_id="", feats=None, head=None, hcfg=None):
    """Evaluasi RM-c untuk SATU config (alpha, k, weighting); log 1 baris; return row.

    `head`/`hcfg` diterima sudah dimuat (bukan reload per config) -- penting untuk grid
    alpha x k yang bisa berisi puluhan-ratusan kombinasi.
    """
    cfg = {**T.RMC_DEFAULT, **cfg_input}
    rid = logger.next_id()
    head_model = hcfg.get("_encoder_model_name")
    if head_model is not None and head_model != ctx["model_name"]:
        raise ValueError(
            f"Encoder mismatch: head RM-b dilatih dengan '{head_model}' tapi job ini pakai "
            f"'{ctx['model_name']}'. RM-c mewarisi encoder dari head RM-b -- jalankan RM-b dulu "
            f"dengan model_name yang sama, atau ganti model_name job ini."
        )
    print(f"[rmc] RUN #{rid} config={cfg} | head RM-b: {hcfg}", flush=True)
    vm, extra = T.eval_rmc(cfg, feats, head, ctx["device"], split="val")
    row = {"scenario": "rmc", "batch_id": batch_id, "model_name": ctx["model_name"], **cfg,
           "head_arch_rmb": hcfg.get("head_arch"),
           "val_f1_macro": round(vm["f1_macro"], 6), "val_acc": round(vm["accuracy"], 6),
           "val_precision_macro": round(vm["precision_macro"], 6),
           "val_recall_macro": round(vm["recall_macro"], 6),
           "val_f1_judi": round(vm["f1_class1"], 6),
           "index_vectors": extra["index_vectors"], "eval_time_s": extra["eval_time_s"],
           "trainable_params": 0, "train_time_s": 0.0, "catatan": note}
    if eval_test:
        tm, ex2 = T.eval_rmc(cfg, feats, head, ctx["device"], split="test")
        row.update({f"test_{k}": round(v, 6) for k, v in tm.items()
                    if k in ["f1_macro", "accuracy", "f1_class1", "precision_class1", "recall_class1"]})
        E.plot_confusion_matrix(feats["test"][1], ex2["preds"],
                                out / "figures" / f"rmc_run{rid}_confusion.png",
                                title=f"RM-c run#{rid} (test)")
        E.plot_pr_curve(feats["test"][1], ex2["p_judi"], out / "figures" / f"rmc_run{rid}_pr.png",
                        title=f"RM-c run#{rid} (test)")
        print(f"[rmc] TEST F1-macro {tm['f1_macro']:.4f}", flush=True)
    row = logger.log(row)
    if update_best(out, "rmc", {"run_id": rid, "config": cfg, "val_f1_macro": vm["f1_macro"],
                                "train_time_s": 0.0, "trainable_params": 0}):
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save({"config": cfg, "run_id": rid, "val_f1_macro": vm["f1_macro"]},
                   out / "checkpoints" / "rmc_best.pt")
    print(f"[rmc] RUN #{rid} val F1-macro {vm['f1_macro']:.4f}", flush=True)
    return row


def phase_rmc(job, ctx, out):
    feats = ensure_features(ctx, out)
    if not (out / "checkpoints" / "rmb_best.pt").exists():
        raise FileNotFoundError("RM-b belum ada (checkpoints/rmb_best.pt). Jalankan RM-b dulu.")
    head, hcfg = load_best_head(out, ctx["device"])
    logger = T.RunLogger(out / "runs_rmc.csv")
    write_progress(out, phase="rmc", status="running", message=f"RM-c run #{logger.next_id()}")
    _run_one_rmc(job.get("config", {}), job.get("note", ""), bool(job.get("eval_test")), ctx, out, logger,
                feats=feats, head=head, hcfg=hcfg)


# ----------------------------- mode batch (banyak config, 1 aksi) -----------------------------

def _log_batch_error(out: Path, phase: str, batch_id: str, seq: int, cfg: dict, note: str, exc: Exception):
    """Catat 1 config yang gagal ke runs_{phase}_errors.csv -- batch TETAP lanjut ke config
    berikutnya; baris yang sudah berhasil di runs_{phase}.csv tidak tersentuh."""
    p = out / f"runs_{phase}_errors.csv"
    rows = pd.read_csv(p).to_dict("records") if p.exists() else []
    rows.append({"seq": seq, "batch_id": batch_id, "config": json.dumps(cfg), "catatan": note,
                 "error_type": type(exc).__name__, "error_message": str(exc)[:500],
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    pd.DataFrame(rows).to_csv(p, index=False)


def run_batch(job, ctx, out):
    """Jalankan job['configs'] (list) secara berurutan dalam SATU subprocess.

    Reuse SATU RunLogger + (untuk rmb/rmc) SATU feats/head resolusi -- bukan per config,
    penting untuk grid RM-c yang bisa berisi puluhan-ratusan kombinasi alpha x k.
    Kegagalan 1 config diisolasi (lihat _log_batch_error); stop.flag memungkinkan
    berhenti graceful setelah config yang sedang berjalan selesai + ter-log.
    """
    phase = job["phase"]
    if phase not in ("rma", "rmb", "rmc"):
        raise ValueError(f"phase '{phase}' tidak mendukung mode batch")
    logger = T.RunLogger(out / f"runs_{phase}.csv")
    configs = job["configs"]
    total = len(configs)
    batch_id = job.get("batch_id") or time.strftime(f"{phase}_grid_%Y%m%d_%H%M%S")
    stop_flag = out / "stop.flag"
    if stop_flag.exists():
        stop_flag.unlink()

    feats = None; head = hcfg = None
    if phase == "rmb":
        feats = ensure_features(ctx, out)
    elif phase == "rmc":
        feats = ensure_features(ctx, out)
        if not (out / "checkpoints" / "rmb_best.pt").exists():
            raise FileNotFoundError("RM-b belum ada (checkpoints/rmb_best.pt). Jalankan RM-b dulu.")
        head, hcfg = load_best_head(out, ctx["device"])

    def runner(cfg_input, note, eval_test, bid):
        if phase == "rma":
            return _run_one_rma(cfg_input, note, eval_test, ctx, out, logger, bid)
        if phase == "rmb":
            return _run_one_rmb(cfg_input, note, eval_test, ctx, out, logger, bid, feats=feats)
        return _run_one_rmc(cfg_input, note, eval_test, ctx, out, logger, bid,
                            feats=feats, head=head, hcfg=hcfg)

    times = []
    last_write = 0.0
    for i, item in enumerate(configs, start=1):
        if stop_flag.exists():
            stop_flag.unlink()
            write_progress(out, status="stopped", phase=phase, batch_id=batch_id, batch_total=total,
                           batch_done=i - 1,
                           message=f"Berhenti diminta -- {i - 1}/{total} config selesai.")
            break
        now = time.perf_counter()
        if i in (1, total) or now - last_write >= 1.0:
            avg = sum(times) / len(times) if times else None
            eta = avg * (total - i + 1) if avg else None
            write_progress(out, phase=phase, status="running", batch_id=batch_id, batch_total=total,
                           batch_done=i - 1, batch_current_config=item.get("config", {}),
                           batch_eta_s=round(eta, 1) if eta else None,
                           message=f"{phase.upper()} batch {i}/{total}")
            last_write = now
        t0 = time.perf_counter()
        try:
            row = runner(item.get("config", {}), item.get("note", ""), bool(item.get("eval_test")), batch_id)
            print(f"[batch {phase}] {i}/{total} OK -> run #{row['run_id']}", flush=True)
        except Exception as e:
            print(f"[batch {phase}] {i}/{total} GAGAL: {e}", flush=True)
            _log_batch_error(out, phase, batch_id, i, item.get("config", {}), item.get("note", ""), e)
            if ctx["device"].type == "cuda":
                torch.cuda.empty_cache()
        times.append(time.perf_counter() - t0)
    else:
        write_progress(out, status="done", phase=phase, batch_id=batch_id, batch_total=total,
                       batch_done=total, message=f"Batch selesai: {total}/{total} config diproses.")

    try:
        reporting.grid_pivot_and_heatmap(out, phase)
        reporting.tradeoff_scatter(out, phase)
        reporting.top_configs_bar_chart(out, phase)
    except Exception as e:
        print(f"[batch] gagal membuat artefak grid (diabaikan): {e}", flush=True)


def phase_final(job, ctx, out):
    """Benchmark inferensi RM-a/b/c SATU SESI (GPU sama) + verdict kriteria sukses.

    Memakai config TERBAIK-sejauh-ini tiap skenario (best.json / checkpoints).
    """
    import rac
    from torch.utils.data import DataLoader
    write_progress(out, phase="final", status="running", message="benchmark inferensi...")
    device = ctx["device"]
    feats = ensure_features(ctx, out)
    best = read_json(out / "best.json")
    for need in ["rma", "rmb", "rmc"]:
        if need not in best:
            raise RuntimeError(f"Skenario '{need}' belum punya run. Jalankan dulu sebelum Final.")

    ck_a = torch.load(out / "checkpoints" / "rma_best.pt", map_location=device)
    model = M.build_finetune_model(ctx["tokenizer"], model_name=ctx["model_name"]).to(device)
    model.load_state_dict(ck_a["model_state"]); model.eval()
    head, hcfg = load_best_head(out, device); head.to(device).eval()
    encoder = M.build_encoder(ctx["tokenizer"], model_name=ctx["model_name"]).to(device)
    ccfg = torch.load(out / "checkpoints" / "rmc_best.pt", map_location=device)["config"]
    index, _ = rac.build_faiss_index(feats["train"][0])

    # --- test metrics untuk ketiganya (sekali, pada config terbaik) ---
    tm_a, yt, pt, prt_a = T.test_rma(model, ctx, ck_a["config"].get("micro_batch", 32))
    E.plot_confusion_matrix(yt, pt, out / "figures" / "final_rma_confusion.png", title="RM-a final (test)")
    E.plot_pr_curve(yt, prt_a, out / "figures" / "final_rma_pr.png", title="RM-a final (test)")
    with torch.no_grad():
        pb_te = rac.softmax(head(torch.tensor(feats["test"][0], device=device)).cpu().numpy())
    tm_b = E.classification_metrics(feats["test"][1], pb_te.argmax(1))
    E.plot_confusion_matrix(feats["test"][1], pb_te.argmax(1), out / "figures" / "final_rmb_confusion.png",
                            title="RM-b final (test)")
    E.plot_pr_curve(feats["test"][1], pb_te[:, 1], out / "figures" / "final_rmb_pr.png",
                    title="RM-b final (test)")
    tm_c, ex_c = T.eval_rmc(ccfg, feats, head, device, split="test")
    E.plot_confusion_matrix(feats["test"][1], ex_c["preds"], out / "figures" / "final_rmc_confusion.png",
                            title="RM-c final (test)")
    E.plot_pr_curve(feats["test"][1], ex_c["p_judi"], out / "figures" / "final_rmc_pr.png",
                    title="RM-c final (test)")

    # --- benchmark inferensi (GPU sama, satu sesi) ---
    ds = GamblingCommentDataset(ctx["test_df"]["text_clean"], ctx["test_df"]["label"],
                                tokenizer=ctx["tokenizer"], max_length=ctx["max_length"])
    one = next(iter(DataLoader(ds, batch_size=1)))
    ids1 = one["input_ids"].to(device); attn1 = one["attention_mask"].to(device)
    tti1 = one.get("token_type_ids"); tti1 = tti1.to(device) if tti1 is not None else None
    qe = feats["test"][0][:1]

    def infer_rma(_):
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            return model(input_ids=ids1, attention_mask=attn1, token_type_ids=tti1).logits

    def infer_rmb(_):
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            o = encoder(input_ids=ids1, attention_mask=attn1, token_type_ids=tti1)
            return head(M.mean_pool(o.last_hidden_state, attn1).float())

    def infer_rmc(_):
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            o = encoder(input_ids=ids1, attention_mask=attn1, token_type_ids=tti1)
            pbq = rac.softmax(head(M.mean_pool(o.last_hidden_state, attn1).float()).cpu().numpy())
        s, i = rac.retrieve(index, qe, int(ccfg["k"]))
        pr = rac.retrieval_distribution(s, feats["train"][1][i], 2, weighting=ccfg.get("weighting", "similarity"))
        return rac.fuse(pbq, pr, float(ccfg["alpha"]))

    bench = []
    for name, fn in [("RM-a", infer_rma), ("RM-b", infer_rmb), ("RM-c", infer_rmc)]:
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        lat = E.measure_latency(fn, None, n_warmup=10, n_runs=100)
        mem = E.peak_gpu_mem_mb()
        bench.append({"scenario": name, "infer_latency_ms": round(lat, 4),
                      "infer_peak_gpu_mem_mb": round(mem, 1)})
        print(f"[final] {name}: {lat:.3f} ms/sample | peak {mem:.0f} MB", flush=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bench).to_csv(out / "metrics" / "inference_benchmark.csv", index=False)
    try:
        reporting.final_inference_bar_chart(out)
    except Exception as e:
        print(f"[final] gagal membuat bar chart inferensi (diabaikan): {e}", flush=True)

    # --- perbandingan + kriteria sukses ---
    rows = []
    for name, key, tm in [("RM-a", "rma", tm_a), ("RM-b", "rmb", tm_b), ("RM-c", "rmc", tm_c)]:
        b = best[key]
        rows.append({"model": name, "config": json.dumps(b["config"]),
                     "test_f1_macro": round(tm["f1_macro"], 6), "test_acc": round(tm["accuracy"], 6),
                     "test_f1_judi": round(tm["f1_class1"], 6),
                     "test_precision_judi": round(tm["precision_class1"], 6),
                     "test_recall_judi": round(tm["recall_class1"], 6),
                     "val_f1_macro": round(b["val_f1_macro"], 6),
                     "trainable_params": b.get("trainable_params"),
                     "train_time_s": b.get("train_time_s")})
    comp = pd.DataFrame(rows).merge(pd.DataFrame(bench).rename(columns={"scenario": "model"}),
                                    on="model", how="left")
    comp.to_csv(out / "metrics" / "final_comparison.csv", index=False)
    print("[final] perbandingan:\n" + comp.to_string(index=False), flush=True)

    a_f1 = tm_a["f1_macro"]; a_par = best["rma"].get("trainable_params"); a_t = best["rma"].get("train_time_s")
    crit = []
    for name, key, tm in [("RM-b", "rmb", tm_b), ("RM-c", "rmc", tm_c)]:
        b = best[key]
        gap = (a_f1 - tm["f1_macro"]) * 100
        p_red = (1 - (b.get("trainable_params") or 0) / a_par) * 100 if a_par else None
        row = {"model": name, "f1_gap_pp": round(gap, 2), "lolos_f1_gap<=3pp": bool(gap <= 3)}
        if p_red is not None:
            row.update({"param_reduction_pct": round(p_red, 4), "lolos_param>=90%": bool(p_red >= 90)})
        if a_t and b.get("train_time_s") is not None:
            t_red = (1 - b["train_time_s"] / a_t) * 100
            row.update({"train_time_s": b["train_time_s"], "rma_train_time_s": a_t,
                        "time_reduction_pct": round(t_red, 2), "lolos_waktu>=50%": bool(t_red >= 50)})
        n = sum(1 for k in ["lolos_f1_gap<=3pp", "lolos_param>=90%", "lolos_waktu>=50%"] if row.get(k))
        row["kriteria_terpenuhi"] = f"{n}/3"; row["kompetitif(>=2/3)"] = bool(n >= 2)
        crit.append(row)
    pd.DataFrame(crit).to_csv(out / "metrics" / "success_criteria.csv", index=False)
    print("[final] kriteria sukses:\n" + pd.DataFrame(crit).to_string(index=False), flush=True)


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    job = json.load(open(args.config, encoding="utf-8"))
    out = ROOT / job.get("out_dir", "results/vast")
    for sub in ["history", "figures", "checkpoints", "metrics"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = write_hardware(out)
    phase = job.get("phase", "rma")
    is_batch = bool(job.get("configs")) and phase in ("rma", "rmb", "rmc")
    # Reset eksplisit field batch_* di setiap job baru -- write_progress() cuma MERGE,
    # tak pernah membersihkan, jadi tanpa ini sisa batch_total/batch_done dari job
    # SEBELUMNYA bisa nyangkut & tampil salah di job yang baru mulai.
    write_progress(out, status="running", phase=phase,
                   started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   message=f"mulai '{phase}' di {hw['gpu']}", error="",
                   batch_id=job.get("batch_id") if is_batch else None,
                   batch_total=len(job.get("configs", [])) if is_batch else None,
                   batch_done=0 if is_batch else None,
                   batch_eta_s=None, batch_current_config=None)
    print(f"=== job_runner: phase={phase} | {hw['gpu']} | torch {hw['torch']} ===", flush=True)
    t0 = time.perf_counter()
    try:
        ctx = build_ctx(job, device)
        if is_batch:
            run_batch(job, ctx, out)
        else:
            {"rma": phase_rma, "rmb": phase_rmb, "rmc": phase_rmc, "final": phase_final}[phase](job, ctx, out)
        cur = read_json(out / "progress.json")
        if cur.get("status") != "stopped":  # jangan timpa status berhenti-graceful
            write_progress(out, status="done", message=f"selesai dalam {time.perf_counter()-t0:.0f}s")
        try:
            reporting.write_run_bundle_summary(out)
        except Exception as e:
            print(f"[main] gagal menulis tuning_summary.json (diabaikan): {e}", flush=True)
        print(f"=== SELESAI ({time.perf_counter()-t0:.0f}s) ===", flush=True)
    except Exception as e:
        write_progress(out, status="error", error=f"{type(e).__name__}: {e}", message="job gagal")
        print("=== ERROR ===\n" + traceback.format_exc(), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
