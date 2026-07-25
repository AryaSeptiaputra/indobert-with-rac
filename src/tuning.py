"""Engine training & evaluasi satu-konfigurasi (single-config) untuk RM-a/RM-b/RM-c.

Alur kerja penelitian ini adalah **human-in-the-loop**, bukan pencarian otomatis:

    set nilai hyperparameter -> jalankan 1 run -> analisis hasil -> ubah nilai -> ulang

Jadi modul ini TIDAK memuat algoritma pencarian (grid/line-search). Satu panggilan =
satu konfigurasi = satu baris di riwayat run. Alasan tiap keputusan dicatat manual di
kolom `catatan` (diisi lewat UI), sehingga tiap nilai HP punya justifikasi eksplisit.

Seleksi HP memakai **validation**; TEST hanya dievaluasi bila diminta eksplisit
(hindari memilih HP berdasarkan test).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from dataset import GamblingCommentDataset
import evaluate as E
import modeling as M

# Nilai awal yang disarankan (titik mulai loop; semuanya bisa diubah dari UI).
# Justifikasi teoretis: Devlin+2019 (grid BERT lr 2-5e-5, batch 16/32),
# Loshchilov & Hutter (AdamW weight_decay), warmup 10% standar BERT/RoBERTa.
RMA_DEFAULT = {"lr": 2e-5, "epochs": 5, "batch": 16, "warmup_ratio": 0.1,
               "weight_decay": 0.01, "micro_batch": 32, "seed": 42}
# Peters+2019 (feature-based transfer: head acak -> lr lebih tinggi, butuh lebih banyak epoch)
RMB_DEFAULT = {"head_arch": "linear", "hidden_dim": 256, "epochs": 5, "lr": 2e-4,
               "dropout": 0.1, "weight_decay": 0.01, "batch": 32, "seed": 42}
# Yu+2023 (RAC): alpha membobot cabang parametrik vs retrieval; k = jumlah tetangga
RMC_DEFAULT = {"alpha": 0.3, "k": 5, "weighting": "similarity"}

# Ambang "seri" (RMA_TUNING_GRID.md: aturan seleksi #2) -> titik tengah pita 0.1-0.2pp.
TIE_THRESHOLD_PP = 0.15


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class RunLogger:
    """Riwayat run yang MENUMPUK (append) lintas sesi — bukan menimpa.

    Tiap run = satu baris: run_id, semua hyperparameter, metrik val (+test bila ada),
    waktu, dan `catatan` (alasan Anda memilih nilai tsb).
    """

    def __init__(self, csv_path):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if self.path.exists():
            try:
                self.rows = pd.read_csv(self.path).to_dict("records")
            except Exception:
                self.rows = []

    def next_id(self) -> int:
        return len(self.rows) + 1

    def log(self, row: dict) -> dict:
        """Tambah 1 baris. Sisipkan kolom turunan (delta/tie/overfit) sebelum menyimpan.

        `delta_vs_best_f1_macro_pp`/`is_tie_with_best`: operasionalisasi aturan seleksi
        di RMA_TUNING_GRID.md (tie <= TIE_THRESHOLD_PP dari juara SEBELUM baris ini).
        `overfit_signal`: best_epoch < epochs -> epoch optimal bukan epoch terakhir.
        """
        if "val_f1_macro" in row and self.rows:
            prior_best = max((r.get("val_f1_macro", -1) for r in self.rows), default=-1)
            if prior_best > -1:
                delta_pp = (row["val_f1_macro"] - prior_best) * 100
                row = {**row, "delta_vs_best_f1_macro_pp": round(delta_pp, 4),
                       "is_tie_with_best": bool(abs(delta_pp) <= TIE_THRESHOLD_PP)}
        if "best_epoch" in row and "epochs" in row:
            row = {**row, "overfit_signal": bool(row["best_epoch"] < row["epochs"])}
        row = {"run_id": self.next_id(), **row,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        return row


# =========================== RM-a ===========================

def _loader(df, bs, shuffle, ctx):
    ds = GamblingCommentDataset(df["text_clean"], df["label"], tokenizer=ctx["tokenizer"],
                                max_length=ctx["max_length"])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=ctx.get("num_workers", 0), pin_memory=True)


@torch.no_grad()
def _predict(model, dl, device):
    """Return (gts, preds, probs_class1) -- probabilitas kelas judi dari softmax logits,
    dipakai untuk PR curve tanpa perlu forward pass tambahan."""
    model.eval(); preds, gts, probs1 = [], [], []
    for b in dl:
        ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device)
        tti = b.get("token_type_ids"); tti = tti.to(device) if tti is not None else None
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=attn, token_type_ids=tti).logits
        p1 = torch.softmax(logits.float(), dim=-1)[:, 1]
        preds.append(logits.argmax(1).cpu().numpy()); gts.append(b["labels"].numpy())
        probs1.append(p1.cpu().numpy())
    return np.concatenate(gts), np.concatenate(preds), np.concatenate(probs1)


def train_eval_rma(cfg, ctx, keep_best=True):
    """Latih RM-a untuk SATU config. Kembalikan (val_metrics_best, history, extra).

    Checkpoint terbaik (by val F1-macro) disimpan di extra['best_state'].
    micro_batch membatasi VRAM: batch efektif = micro_batch x grad_accum.
    """
    set_seed(cfg["seed"]); device = ctx["device"]
    model = M.build_finetune_model(ctx["tokenizer"], model_name=ctx["model_name"]).to(device)
    micro = min(cfg["batch"], cfg.get("micro_batch", 16))
    accum = max(1, cfg["batch"] // micro)
    tr = _loader(ctx["train_df"], micro, True, ctx); va = _loader(ctx["val_df"], micro, False, ctx)
    crit = nn.CrossEntropyLoss(weight=ctx["weight"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = math.ceil(len(tr) / accum) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    best_f1, best_m, best_ep, best_state, hist = -1.0, None, 0, None, []
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); opt.zero_grad()
        run_loss = 0.0
        for i, b in enumerate(tr):
            ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device)
            tti = b.get("token_type_ids"); tti = tti.to(device) if tti is not None else None
            y = b["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=attn, token_type_ids=tti).logits
                loss = crit(logits, y) / accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * accum * y.size(0)
            if (i + 1) % accum == 0 or (i + 1) == len(tr):
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        yv, pv, _ = _predict(model, va, device); vm = E.classification_metrics(yv, pv)
        hist.append({"epoch": ep, "train_loss": round(run_loss / len(ctx["train_df"]), 6),
                     "val_f1_macro": vm["f1_macro"], "val_acc": vm["accuracy"],
                     "val_f1_judi": vm["f1_class1"], "val_precision_judi": vm["precision_class1"],
                     "val_recall_judi": vm["recall_class1"]})
        print(f"  [RM-a] epoch {ep}/{cfg['epochs']} val F1-macro {vm['f1_macro']:.4f}", flush=True)
        if vm["f1_macro"] > best_f1:
            best_f1, best_m, best_ep = vm["f1_macro"], vm, ep
            if keep_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    extra = {"best_epoch": best_ep, "train_time_s": round(time.perf_counter() - t0, 1),
             "peak_mem_mb": round(E.peak_gpu_mem_mb(), 1), "best_state": best_state,
             "micro_batch": micro, "grad_accum": accum, "model": model,
             "trainable_params": E.count_parameters(model)["trainable_params"]}
    return best_m, hist, extra


def test_rma(model, ctx, micro_batch=32):
    yt, pt, prt = _predict(model, _loader(ctx["test_df"], micro_batch, False, ctx), ctx["device"])
    return E.classification_metrics(yt, pt), yt, pt, prt


# =========================== RM-b ===========================

def train_eval_rmb(cfg, feats, weight, device, return_test=False):
    """Latih head RM-b (linear/mlp) di atas fitur beku — SATU config."""
    set_seed(cfg["seed"])
    H = feats["train"][0].shape[1]
    head = M.build_head(cfg["head_arch"], hidden_size=H, dropout=cfg["dropout"],
                        hidden_dim=cfg.get("hidden_dim", 256)).to(device)
    Xtr = torch.tensor(feats["train"][0], device=device); ytr = torch.tensor(feats["train"][1], device=device)
    Xva = torch.tensor(feats["val"][0], device=device); yva = feats["val"][1]
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg["batch"], shuffle=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    best_f1, best_m, best_ep, best_state, hist = -1.0, None, 0, None, []
    for ep in range(1, cfg["epochs"] + 1):
        head.train()
        run_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad(); loss = crit(head(xb), yb); loss.backward(); opt.step()
            run_loss += loss.item() * xb.size(0)
        head.eval()
        with torch.no_grad():
            pv = head(Xva).argmax(1).cpu().numpy()
        vm = E.classification_metrics(yva, pv)
        hist.append({"epoch": ep, "train_loss": round(run_loss / len(Xtr), 6),
                     "val_f1_macro": vm["f1_macro"], "val_acc": vm["accuracy"],
                     "val_f1_judi": vm["f1_class1"], "val_precision_judi": vm["precision_class1"],
                     "val_recall_judi": vm["recall_class1"]})
        if vm["f1_macro"] > best_f1:
            best_f1, best_m, best_ep = vm["f1_macro"], vm, ep
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    train_time_s = round(time.perf_counter() - t0, 2)
    peak_mem_mb = round(E.peak_gpu_mem_mb(), 1)
    head.eval()
    infer_latency_ms = E.measure_latency(lambda _: head(Xva[:1]), None, n_warmup=5, n_runs=50)
    extra = {"best_epoch": best_ep, "train_time_s": train_time_s,
             "trainable_params": E.count_parameters(head)["trainable_params"],
             "best_state": best_state, "hidden_size": H,
             "peak_mem_mb": peak_mem_mb, "infer_latency_ms": round(infer_latency_ms, 4)}
    if return_test:
        head.load_state_dict(best_state); head.to(device).eval()
        with torch.no_grad():
            logits_te = head(torch.tensor(feats["test"][0], device=device))
            pte = logits_te.argmax(1).cpu().numpy()
            prte = torch.softmax(logits_te.float(), dim=-1)[:, 1].cpu().numpy()
        extra["test_metrics"] = E.classification_metrics(feats["test"][1], pte)
        extra["test_pred"] = pte
        extra["test_pred_proba"] = prte
    return best_m, hist, extra


# =========================== RM-c ===========================

def eval_rmc(cfg, feats, head, device, split="val"):
    """Evaluasi RM-c (RAC) untuk SATU config (alpha, k, weighting). Tanpa training.

    p_final = (1-alpha)*softmax(head(emb)) + alpha*p_retr ; index FAISS HANYA dari train.
    """
    import rac
    head.eval().to(device)
    with torch.no_grad():
        pb = rac.softmax(head(torch.tensor(feats[split][0], device=device)).cpu().numpy())
    index, _ = rac.build_faiss_index(feats["train"][0])
    t0 = time.perf_counter()
    preds, p_final = rac.rac_predict(index, feats["train"][1], feats[split][0], pb,
                                     k=int(cfg["k"]), alpha=float(cfg["alpha"]),
                                     weighting=cfg.get("weighting", "similarity"))
    m = E.classification_metrics(feats[split][1], preds)
    extra = {"eval_time_s": round(time.perf_counter() - t0, 2), "index_vectors": int(index.ntotal),
             "preds": preds, "p_judi": p_final[:, 1]}
    return m, extra
