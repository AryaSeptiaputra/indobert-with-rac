"""Pelatihan ulang RM-a & RM-b di GPU LOKAL + rerun RM-c, dalam SATU sesi.

Tujuan: reproduksi lokal + pengukuran efisiensi apple-to-apple (latency & peak
memory inferensi ketiga model pada GPU yang sama). Hasil ditulis ke
`results/local/` agar **tidak menimpa** hasil Colab di `results/`.

Windows-safe: DataLoader num_workers=0. VRAM-safe: coba bs=16, fallback bs=8 saat OOM.
Jalankan:  .venv\\Scripts\\python.exe run_local_training.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys, json, time, random
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from dataset import load_tokenizer, GamblingCommentDataset
import modeling as M
import evaluate as E
import rac

SEED = 42
MODEL_NAME = "indobenchmark/indobert-base-p2"
MAXLEN = 128

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

DATA = Path("dataset/splits"); META = Path("dataset/processed/metadata.json")
OUT = Path("results/local")
for sub in ["checkpoints/rma", "checkpoints/rmb", "features", "metrics", "figures"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA / "train.csv")
val_df   = pd.read_csv(DATA / "val.csv")
test_df  = pd.read_csv(DATA / "test.csv")
cw = json.load(open(META, encoding="utf-8"))["class_weights"]
weight = torch.tensor([cw["0"], cw["1"]], dtype=torch.float, device=device)
tokenizer = load_tokenizer(MODEL_NAME)
print(f"data: {len(train_df)}/{len(val_df)}/{len(test_df)} | class weights {weight.tolist()}")

def loader(df, bs, shuffle):
    ds = GamblingCommentDataset(df["text_clean"], df["label"], tokenizer=tokenizer, max_length=MAXLEN)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=True)

def move(batch):
    tti = batch.get("token_type_ids")
    return (batch["input_ids"].to(device), batch["attention_mask"].to(device),
            tti.to(device) if tti is not None else None)

@torch.no_grad()
def predict_model(model, dl):
    model.eval(); preds, gts = [], []
    for b in dl:
        ids, attn, tti = move(b)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=attn, token_type_ids=tti).logits
        preds.append(logits.argmax(1).cpu().numpy()); gts.append(b["labels"].numpy())
    return np.concatenate(gts), np.concatenate(preds)

# ============================ RM-a ============================
def train_rma(bs):
    set_seed()
    model = M.build_finetune_model(tokenizer, model_name=MODEL_NAME).to(device)
    tr = loader(train_df, bs, True); va = loader(val_df, bs, False)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = len(tr) * 5
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    hist, best = [], -1.0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for ep in range(1, 6):
        model.train(); run = 0.0; te = time.perf_counter()
        for b in tr:
            ids, attn, tti = move(b); y = b["labels"].to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=attn, token_type_ids=tti).logits
                loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item() * y.size(0)
        yv, pv = predict_model(model, va); vm = E.classification_metrics(yv, pv)
        hist.append({"epoch": ep, "train_loss": run/len(train_df), "val_f1_macro": vm["f1_macro"],
                     "val_acc": vm["accuracy"], "epoch_time_s": time.perf_counter()-te})
        print(f"  [RM-a] ep{ep} loss {run/len(train_df):.4f} val F1 {vm['f1_macro']:.4f} ({hist[-1]['epoch_time_s']:.0f}s)")
        if vm["f1_macro"] > best:
            best = vm["f1_macro"]
            torch.save({"model_state": model.state_dict(), "epoch": ep, "val_f1_macro": best},
                       OUT / "checkpoints/rma/best.pt")
    total_t = time.perf_counter() - t0; peak = E.peak_gpu_mem_mb()
    ck = torch.load(OUT / "checkpoints/rma/best.pt", map_location=device)
    model.load_state_dict(ck["model_state"])
    yt, pt = predict_model(model, loader(test_df, bs, False))
    m = E.classification_metrics(yt, pt)
    E.plot_confusion_matrix(yt, pt, OUT / "figures/rma_confusion.png", title="RM-a (lokal) — Confusion")
    par = E.count_parameters(model)
    row = {"scenario": "RM-a", **{k: round(v, 6) for k, v in m.items()},
           "trainable_params": par["trainable_params"], "batch_size": bs,
           "total_train_time_s": round(total_t, 1), "sec_per_epoch": round(total_t/5, 1),
           "train_peak_gpu_mem_mb": round(peak, 1), "best_val_f1_macro": round(best, 6)}
    E.save_metrics_csv(row, OUT / "metrics/rma_metrics.csv")
    pd.DataFrame(hist).to_csv(OUT / "metrics/rma_history.csv", index=False)
    return model, row

print("\n=== RM-a (full fine-tuning) LOKAL ===")
try:
    rma_model, rma_row = train_rma(16)
    used_bs = 16
except torch.cuda.OutOfMemoryError:
    print("  OOM pada bs=16 -> coba bs=8"); torch.cuda.empty_cache()
    rma_model, rma_row = train_rma(8); used_bs = 8
print("RM-a:", {k: rma_row[k] for k in ["f1_macro","trainable_params","total_train_time_s","train_peak_gpu_mem_mb"]})

# ============================ RM-b ============================
print("\n=== RM-b (frozen encoder + head) LOKAL ===")
set_seed()
encoder = M.build_encoder(tokenizer, model_name=MODEL_NAME).to(device)
feats = {}
if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
t_ext = time.perf_counter()
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    emb, lab = M.extract_features(encoder, loader(df, 32, False), device, use_amp=True)
    feats[name] = (emb, lab)
    np.save(OUT / f"features/{name}_emb.npy", emb); np.save(OUT / f"features/{name}_label.npy", lab)
ext_t = time.perf_counter() - t_ext; ext_peak = E.peak_gpu_mem_mb()
print(f"  ekstraksi {ext_t:.0f}s peak {ext_peak:.0f}MB")

set_seed()
Xtr = torch.tensor(feats["train"][0], device=device); ytr = torch.tensor(feats["train"][1], device=device)
Xva = torch.tensor(feats["val"][0], device=device); yva = feats["val"][1]
head = M.FrozenHead(hidden_size=Xtr.shape[1]).to(device)
crit = nn.CrossEntropyLoss(weight=weight)
opt = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=0.01)
hl = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
best = -1.0; t_head = time.perf_counter()
for ep in range(1, 6):
    head.train()
    for xb, yb in hl:
        opt.zero_grad(); loss = crit(head(xb), yb); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad(): pv = head(Xva).argmax(1).cpu().numpy()
    f1 = E.classification_metrics(yva, pv)["f1_macro"]
    print(f"  [RM-b] ep{ep} val F1 {f1:.4f}")
    if f1 > best:
        best = f1
        torch.save({"head_state": head.state_dict(), "hidden_size": Xtr.shape[1], "val_f1_macro": best},
                   OUT / "checkpoints/rmb/head_best.pt")
head_t = time.perf_counter() - t_head
ckb = torch.load(OUT / "checkpoints/rmb/head_best.pt", map_location=device)
head.load_state_dict(ckb["head_state"]); head.eval()
Xte = torch.tensor(feats["test"][0], device=device); yte = feats["test"][1]
with torch.no_grad(): pte = head(Xte).argmax(1).cpu().numpy()
mb = E.classification_metrics(yte, pte)
E.plot_confusion_matrix(yte, pte, OUT / "figures/rmb_confusion.png", title="RM-b (lokal) — Confusion")
rmb_row = {"scenario": "RM-b", **{k: round(v, 6) for k, v in mb.items()},
           "trainable_params": E.count_parameters(head)["trainable_params"],
           "extract_time_s": round(ext_t, 1), "head_train_time_s": round(head_t, 1),
           "total_train_time_s": round(ext_t + head_t, 1),
           "extract_peak_gpu_mem_mb": round(ext_peak, 1), "best_val_f1_macro": round(best, 6)}
E.save_metrics_csv(rmb_row, OUT / "metrics/rmb_metrics.csv")
print("RM-b:", {k: rmb_row[k] for k in ["f1_macro","trainable_params","total_train_time_s"]})

# ============================ RM-c (pada fitur LOKAL) ============================
print("\n=== RM-c (RAC) pada fitur lokal ===")
emb = {s: feats[s][0] for s in ["train","val","test"]}
lab = {s: feats[s][1] for s in ["train","val","test"]}
with torch.no_grad():
    pb = {s: rac.softmax(head(torch.tensor(emb[s], device=device)).cpu().numpy()) for s in ["val","test"]}
index, _ = rac.build_faiss_index(emb["train"])
grid = {}
for k in [1,3,5,10,20,50]:
    sims, idx = rac.retrieve(index, emb["val"], k)
    pr = rac.retrieval_distribution(sims, lab["train"][idx], 2)
    for a in [round(x,2) for x in np.arange(0,1.01,0.1)]:
        grid[(k,a)] = E.classification_metrics(lab["val"], rac.fuse(pb["val"], pr, a).argmax(1))["f1_macro"]
bf, bk, ba = max((v,k,a) for (k,a),v in grid.items())
preds, _ = rac.rac_predict(index, lab["train"], emb["test"], pb["test"], k=bk, alpha=ba)
mc = E.classification_metrics(lab["test"], preds)
E.plot_confusion_matrix(lab["test"], preds, OUT / "figures/rmc_confusion.png", title=f"RM-c (lokal) k={bk} a={ba}")
rmc_row = {"scenario": "RM-c", **{k: round(v,6) for k,v in mc.items()},
           "best_k": bk, "best_alpha": ba, "trainable_params_new": 0, "val_f1_macro_best": round(bf,6)}
E.save_metrics_csv(rmc_row, OUT / "metrics/rmc_metrics.csv")
print(f"RM-c: F1-macro {mc['f1_macro']:.4f} (k={bk}, alpha={ba})")

# ============================ Benchmark inferensi SATU SESI ============================
print("\n=== Benchmark inferensi (GPU sama, satu sesi) ===")
def infer_peak(fn, sample, bs_mem=32):
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    fn(sample); torch.cuda.synchronize() if device.type=="cuda" else None
    return E.peak_gpu_mem_mb()

one = next(iter(loader(test_df, 1, False)))
ids1, attn1, tti1 = move(one)
rma_model.eval()
def infer_rma(_):
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type=="cuda"):
        return rma_model(input_ids=ids1, attention_mask=attn1, token_type_ids=tti1).logits
def infer_rmb(_):
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type=="cuda"):
        o = encoder(input_ids=ids1, attention_mask=attn1, token_type_ids=tti1)
        return head(M.mean_pool(o.last_hidden_state, attn1).float())
qe = emb["test"][:1]
def infer_rmc(_):
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type=="cuda"):
        o = encoder(input_ids=ids1, attention_mask=attn1, token_type_ids=tti1)
        pbq = rac.softmax(head(M.mean_pool(o.last_hidden_state, attn1).float()).cpu().numpy())
    s,i = rac.retrieve(index, qe, bk); pr = rac.retrieval_distribution(s, lab["train"][i], 2)
    return rac.fuse(pbq, pr, ba)

bench = []
for name, fn in [("RM-a", infer_rma), ("RM-b", infer_rmb), ("RM-c", infer_rmc)]:
    lat = E.measure_latency(fn, None, n_warmup=10, n_runs=100)
    mem = infer_peak(fn, None)
    bench.append({"scenario": name, "infer_latency_ms": round(lat,4), "infer_peak_gpu_mem_mb": round(mem,1)})
    print(f"  {name}: {lat:.3f} ms/sample | peak {mem:.0f} MB")
pd.DataFrame(bench).to_csv(OUT / "metrics/inference_benchmark.csv", index=False)

# ============================ Perbandingan lokal vs Colab ============================
def load1(p): return pd.read_csv(p).iloc[0].to_dict() if Path(p).exists() else {}
rows = []
for name, lrow in [("RM-a", rma_row), ("RM-b", rmb_row), ("RM-c", rmc_row)]:
    colab = load1(Path("results/metrics") / f"{name.lower().replace('-','')}_metrics.csv")
    rows.append({"model": name, "f1_macro_local": round(lrow["f1_macro"],4),
                 "f1_macro_colab": round(colab.get("f1_macro"),4) if colab.get("f1_macro") is not None else None})
comp = pd.DataFrame(rows)
comp.to_csv(OUT / "metrics/local_vs_colab.csv", index=False)
print("\n=== Lokal vs Colab (F1-macro) ===")
print(comp.to_string(index=False))
print(f"\nGPU aktual RM-a batch_size = {used_bs}")
print("Semua hasil lokal di: results/local/  (hasil Colab di results/ tetap utuh)")
print("\nSELESAI")
