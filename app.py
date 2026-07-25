"""Control Panel Training & Tuning — IndoBERT-with-RAC (Streamlit).

Alur kerja: **1 klik = 1 konfigurasi**.
    set nilai HP -> Run -> analisis hasil (bareng Claude) -> ubah nilai -> ulang
Tidak ada pencarian otomatis: strategi eksplorasi ada di tangan Anda, dan alasan tiap
keputusan dicatat di kolom "catatan" sehingga tiap nilai HP punya justifikasi eksplisit.

Job dijalankan sebagai PROSES TERPISAH (src/job_runner.py) → tetap jalan meski tab ditutup.

Jalankan:
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Akses:  http://<IP-instance>:<port-eksternal>   (butuh -p 8501:8501 saat sewa)
Plan B: cloudflared tunnel --url http://localhost:8501
"""
import io
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

st.set_page_config(page_title="IndoBERT-RAC — Tuning Panel", page_icon="🎛️", layout="wide")

# ----------------------------- util -----------------------------

def out_dir() -> Path:
    d = ROOT / st.session_state.get("out_dir", "results/vast")
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def job_alive() -> bool:
    """Cek apakah subprocess job_runner.py masih hidup.

    `start_job()` tidak pernah `.wait()` pada Popen-nya, jadi child yang sudah selesai
    jadi zombie -- `os.kill(pid, 0)` TETAP sukses untuk zombie (masih ada di process
    table), sehingga tanpa reap eksplisit fungsi ini bisa melaporkan True selamanya
    walau job sudah lama selesai (bikin panel batch nyangkut & tombol Stop tak berefek).
    `os.waitpid(pid, os.WNOHANG)` non-blocking: me-reap kalau child sudah keluar.
    """
    f = out_dir() / "job.pid"
    if not f.exists():
        return False
    try:
        pid = int(f.read_text().strip())
        try:
            reaped, _ = os.waitpid(pid, os.WNOHANG)
            if reaped == pid:
                return False  # zombie -> baru di-reap -> benar-benar mati
        except ChildProcessError:
            pass  # bukan child proses saat ini (mis. Streamlit sempat direstart)
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def start_job(job: dict):
    if job_alive():
        st.warning("Job lain masih berjalan. Tunggu atau hentikan dulu."); return
    o = out_dir()
    p_cfg = o / "job.json"
    json.dump(job, open(p_cfg, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log = open(o / "job.log", "w", encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen([sys.executable, str(SRC / "job_runner.py"), "--config", str(p_cfg)],
                         stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    (o / "job.pid").write_text(str(p.pid))
    st.success(f"Run '{job['phase']}' dimulai (PID {p.pid}). Aman ditinggal — tutup tab pun jalan terus.")
    time.sleep(1.5); st.rerun()


def stop_job():
    try:
        pid = int((out_dir() / "job.pid").read_text().strip())
        os.kill(pid, signal.SIGTERM); st.warning(f"Stop dikirim ke PID {pid}.")
    except Exception as e:
        st.error(f"Gagal: {e}")


def stop_after_current():
    """Minta job_runner berhenti graceful (di akhir config batch yang sedang berjalan)."""
    (out_dir() / "stop.flag").touch()


def make_zip(paths, base: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(base.parent))
    return buf.getvalue()


def show_runs(name: str, cols_first):
    """Tabel riwayat run (menumpuk) + sorotan best-sejauh-ini."""
    p = out_dir() / f"runs_{name}.csv"
    if not p.exists():
        st.info("Belum ada run. Atur nilai HP di atas lalu klik Run."); return
    d = pd.read_csv(p)
    if "batch_id" in d.columns and d["batch_id"].fillna("").astype(str).ne("").any():
        opts = ["(semua)"] + sorted(x for x in d["batch_id"].dropna().astype(str).unique() if x)
        sel = st.selectbox("Filter batch", opts, key=f"filter_batch_{name}")
        if sel != "(semua)":
            d = d[d["batch_id"].astype(str) == sel]
    st.markdown(f"**Riwayat run ({len(d)}) — menumpuk lintas iterasi**")
    order = [c for c in cols_first if c in d.columns] + [c for c in d.columns if c not in cols_first]
    st.dataframe(d[order], width='stretch', height=300)
    if "val_f1_macro" in d.columns and len(d):
        b = d.loc[d["val_f1_macro"].idxmax()]
        st.success(f"🏆 Terbaik sejauh ini: run #{int(b['run_id'])} — val F1-macro **{b['val_f1_macro']:.4f}**")
        if len(d) > 1:
            st.line_chart(d.set_index("run_id")["val_f1_macro"], height=180)
    err_p = out_dir() / f"runs_{name}_errors.csv"
    if err_p.exists():
        ed = pd.read_csv(err_p)
        if len(ed):
            with st.expander(f"⚠️ {len(ed)} config gagal di batch (lihat detail)"):
                st.dataframe(ed, width='stretch', height=180)


# ----------------------------- sidebar -----------------------------

st.sidebar.title("🎛️ Tuning Panel")
st.session_state.setdefault("out_dir", "results/vast")
st.session_state["out_dir"] = st.sidebar.text_input("Folder output", st.session_state["out_dir"],
                                                    key="sb_out")
smoke = st.sidebar.checkbox("Mode SMOKE (subset kecil, uji cepat)", value=False, key="sb_smoke")
st.sidebar.caption("Alur: set HP → Run → analisis → ubah → ulang. 1 klik = 1 konfigurasi.")

prog = read_json(out_dir() / "progress.json")
alive = job_alive()
st.sidebar.markdown("### Status")
if alive:
    st.sidebar.info(f"🟢 BERJALAN — {prog.get('phase','?')}")
else:
    st.sidebar.markdown({"done": "✅ selesai", "error": "❌ error",
                         "running": "⚠️ terputus?"}.get(prog.get("status", "idle"), "⚪ idle"))
if prog.get("message"):
    st.sidebar.caption(prog["message"])
if prog.get("error"):
    st.sidebar.error(str(prog["error"])[:300])
if alive and prog.get("batch_total"):
    b_done, b_total = prog.get("batch_done", 0), prog["batch_total"]
    b_done = min(b_done, b_total)  # jaga-jaga dari sisa data basi (done tak boleh > total)
    st.sidebar.progress(min(b_done / b_total, 1.0) if b_total else 0.0)
    if b_done >= b_total:
        st.sidebar.caption(f"Batch selesai: {b_total}/{b_total}")
    else:
        b_eta = prog.get("batch_eta_s")
        st.sidebar.caption(f"Batch run {b_done + 1}/{b_total}" +
                           (f" — ETA ~{b_eta / 60:.0f} menit" if b_eta else ""))
        if st.sidebar.button("⏸ Hentikan setelah run ini", key="sb_stop_graceful"):
            stop_after_current(); st.sidebar.info("Akan berhenti setelah config saat ini selesai.")
if alive and st.sidebar.button("⛔ Hentikan paksa (SIGTERM)", key="sb_stop"):
    stop_job(); st.rerun()
auto = st.sidebar.checkbox("Auto-refresh (5 dtk)", value=alive, key="sb_auto")

best = read_json(out_dir() / "best.json")
if best:
    st.sidebar.markdown("### Terbaik sejauh ini")
    for k in ["rma", "rmb", "rmc"]:
        if k in best:
            st.sidebar.write(f"**{k.upper()}**: val F1 {best[k]['val_f1_macro']:.4f} (run #{best[k]['run_id']})")

# Nilai awal widget tuning. Di-seed ke session_state supaya widget TIDAK perlu memakai
# argumen `value=`/`index=`: kombinasi default + key yang sudah ada di session_state
# memicu peringatan Streamlit di tiap rerun.
TUNE_DEFAULTS = {
    "a_lr": 2e-5, "a_ep": 5, "a_bs": 16, "a_wu": 0.1, "a_wd": 0.01, "a_mb": 32,
    "b_arch": "linear", "b_hd": 256, "b_ep": 5, "b_bs": 32,
    "b_lr": 2e-4, "b_do": 0.1, "b_wd": 0.01,
    "c_alpha": 0.3, "c_k": 5, "c_w": "similarity",
}
# Streamlit membuang state widget yang tidak ikut ter-render pada suatu rerun. Karena tab
# Tuning hanya merender satu skenario, tulis-ulang state agar nilai HP yang sudah diisi
# tidak hilang saat berpindah skenario (RM-a -> RM-b -> RM-a).
for _k in list(TUNE_DEFAULTS) + ["a_note", "a_test", "b_note", "b_test", "c_note", "c_test"]:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]
    elif _k in TUNE_DEFAULTS:
        st.session_state[_k] = TUNE_DEFAULTS[_k]

t_st, t_tune, t_f, t_r = st.tabs(["1 Status", "2 Tuning", "3 Final", "4 Hasil & Unduh"])

# ----------------------------- 1 Status -----------------------------
with t_st:
    st.header("Status sistem")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hardware")
        hw = read_json(out_dir() / "hardware.json")
        # Catatan: JANGAN pakai ekspresi ternary sebagai statement di Streamlit —
        # "magic" akan me-render nilai balik (DeltaGenerator) ke halaman.
        if hw:
            st.json(hw)
        else:
            st.caption("hardware.json dibuat otomatis saat run pertama.")
    with c2:
        st.subheader("Data")
        for s in ["train", "val", "test"]:
            p = ROOT / "dataset" / "splits" / f"{s}.csv"
            st.write(f"{'✅' if p.exists() else '❌'} `{s}.csv`" +
                     (f" — {len(pd.read_csv(p)):,} baris" if p.exists() else ""))
        m = ROOT / "dataset" / "processed" / "metadata.json"
        if m.exists():
            st.write("class weights:", read_json(m).get("class_weights"))
        f = out_dir() / "features"
        ok = all((f / f"{s}_emb.npy").exists() for s in ["train", "val", "test"])
        st.write(f"{'✅' if ok else '⚪'} fitur beku (otomatis saat RM-b pertama)")
    st.subheader("Log run (100 baris terakhir)")
    lg = out_dir() / "job.log"
    st.code("".join(open(lg, encoding="utf-8", errors="replace").readlines()[-100:])
            if lg.exists() else "(belum ada)", language="text")

# ----------------------------- 2 Tuning -----------------------------
SCENARIOS = {"rma": "RM-a — Full Fine-tuning",
             "rmb": "RM-b — Frozen Encoder + Head",
             "rmc": "RM-c — RAC (tanpa training)"}
RUN_COLS = {
    "rma": ["run_id", "batch_id", "lr", "epochs", "batch", "warmup_ratio", "weight_decay",
            "val_f1_macro", "val_f1_judi", "is_tie_with_best", "overfit_signal",
            "best_epoch", "train_time_s", "catatan"],
    "rmb": ["run_id", "batch_id", "head_arch", "hidden_dim", "epochs", "lr", "dropout", "weight_decay",
            "val_f1_macro", "val_f1_judi", "is_tie_with_best", "overfit_signal",
            "best_epoch", "train_time_s", "peak_mem_mb", "infer_latency_ms", "catatan"],
    "rmc": ["run_id", "batch_id", "alpha", "k", "weighting", "val_f1_macro", "val_f1_judi",
            "is_tie_with_best", "eval_time_s", "catatan"],
}

# Field HP per skenario yang bisa dijadikan sumbu grid batch (nama, tipe-cast).
FIELD_SPECS = {
    "rma": [("lr", float), ("epochs", int), ("batch", int),
            ("warmup_ratio", float), ("weight_decay", float), ("micro_batch", int)],
    "rmb": [("head_arch", str), ("hidden_dim", int), ("epochs", int), ("batch", int),
            ("lr", float), ("dropout", float), ("weight_decay", float)],
    "rmc": [("alpha", float), ("k", int), ("weighting", str)],
}
# Pemetaan field batch -> key session_state widget single-config yang setara (utk nilai awal).
SCEN_FIELD_KEY = {
    "rma": {"lr": "a_lr", "epochs": "a_ep", "batch": "a_bs", "warmup_ratio": "a_wu",
            "weight_decay": "a_wd", "micro_batch": "a_mb"},
    "rmb": {"head_arch": "b_arch", "hidden_dim": "b_hd", "epochs": "b_ep", "batch": "b_bs",
            "lr": "b_lr", "dropout": "b_do", "weight_decay": "b_wd"},
    "rmc": {"alpha": "c_alpha", "k": "c_k", "weighting": "c_w"},
}
# Kolom waktu di runs_{scenario}.csv dipakai utk estimasi durasi batch + fallback kasar.
BATCH_TIME_COL = {"rma": "train_time_s", "rmb": "head_train_time_s", "rmc": "eval_time_s"}
BATCH_TIME_DEFAULT_S = {"rma": 150.0, "rmb": 5.0, "rmc": 0.5}


def make_batch_note(scenario: str, cfg: dict, axis_fields: list, locked_fields: list,
                    common_reason: str) -> str:
    """Template catatan otomatis (pola tuning_grids/RMA_TUNING_GRID.md) — alasan umum diisi SEKALI oleh
    pengguna, bukan diketik ulang per baris grid."""
    axis_str = " / ".join(f"{k}={cfg[k]}" for k in axis_fields if k in cfg)
    locked_str = ", ".join(f"{k}={cfg[k]}" for k in locked_fields if k in cfg)
    base = f"Grid kombinatorial {scenario}, sel {axis_str}" if axis_str else f"Grid {scenario}"
    base += f"; {locked_str} dikunci." if locked_str else "."
    return f"{base} {common_reason}".strip() if common_reason else base

with t_tune:
    scenario = st.selectbox("Skenario", list(SCENARIOS), format_func=SCENARIOS.get,
                            key="tune_scenario",
                            help="Pilih skenario yang ingin dituning. Riwayat run di bawah "
                                 "mengikuti skenario yang dipilih.")
    st.header(SCENARIOS[scenario])

    if scenario == "rma":
        st.caption("Isi nilai hyperparameter → Run. Hasil dianalisis bersama, lalu ubah nilainya di sini "
                   "dan Run lagi. Titik awal: Devlin+2019 (lr 2e-5, batch 16).")
        c1, c2 = st.columns(2)
        with c1:
            a_lr = st.number_input("learning rate", format="%.8f", step=1e-5, key="a_lr")
            a_ep = st.number_input("epochs", step=1, min_value=1, key="a_ep")
            a_bs = st.number_input("batch (efektif)", step=8, min_value=1, key="a_bs")
        with c2:
            a_wu = st.number_input("warmup_ratio", step=0.05, format="%.2f", key="a_wu")
            a_wd = st.number_input("weight_decay", format="%.4f", step=0.01, key="a_wd")
            a_mb = st.number_input("micro_batch (batas VRAM; 32 di 3090)", step=8, min_value=1,
                                   key="a_mb", help="batch efektif = micro_batch × grad_accum (otomatis)")
        note = st.text_input("Catatan / alasan memilih nilai ini", key="a_note",
                             placeholder="mis. 'baseline Devlin' atau 'naikkan lr krn val masih underfit di run #1'")
        eval_test = st.checkbox("Evaluasi TEST juga", key="a_test",
                                help="⚠️ Pilih HP dari VAL saja. Test sebaiknya dievaluasi di akhir "
                                     "(atau lewat tab Final) agar tidak bias.")
        config = {"lr": a_lr, "epochs": int(a_ep), "batch": int(a_bs),
                  "warmup_ratio": a_wu, "weight_decay": a_wd,
                  "micro_batch": int(a_mb), "seed": 42}

    elif scenario == "rmb":
        st.caption("Encoder beku; hanya head dilatih di atas fitur cache (tiap run hitungan detik). "
                   "Peters+2019: head acak → lr lebih tinggi & butuh lebih banyak epoch dari fine-tuning.")
        c1, c2 = st.columns(2)
        with c1:
            b_arch = st.selectbox("head_arch", ["linear", "mlp"], key="b_arch")
            b_hd = st.number_input("hidden_dim (khusus MLP)", step=64, min_value=8, key="b_hd")
            b_ep = st.number_input("epochs", step=5, min_value=1, key="b_ep")
            b_bs = st.number_input("batch", step=8, min_value=1, key="b_bs")
        with c2:
            b_lr = st.number_input("learning rate", format="%.8f", step=1e-4, key="b_lr")
            b_do = st.number_input("dropout", step=0.1, format="%.2f", key="b_do")
            b_wd = st.number_input("weight_decay", format="%.4f", step=0.01, key="b_wd")
        note = st.text_input("Catatan / alasan", key="b_note",
                             placeholder="mis. 'val F1 masih naik di epoch terakhir → tambah epoch'")
        eval_test = st.checkbox("Evaluasi TEST juga", key="b_test")
        config = {"head_arch": b_arch, "hidden_dim": int(b_hd), "epochs": int(b_ep),
                  "lr": b_lr, "dropout": b_do, "weight_decay": b_wd,
                  "batch": int(b_bs), "seed": 42}

    else:  # rmc
        st.caption("Memakai head RM-b TERBAIK + FAISS (index hanya dari train). "
                   "α=0 → murni head RM-b; α=1 → murni retrieval. Yu+2023.")
        if "rmb" in best:
            st.info(f"Head RM-b yang dipakai: run #{best['rmb']['run_id']} — "
                    f"`{json.dumps(best['rmb']['config'])}` (val F1 {best['rmb']['val_f1_macro']:.4f})")
        else:
            st.warning("Belum ada run RM-b. Jalankan RM-b dulu.")
        c1, c2, c3 = st.columns(3)
        with c1:
            c_alpha = st.number_input("alpha (bobot retrieval)", step=0.1, min_value=0.0,
                                      max_value=1.0, format="%.2f", key="c_alpha")
        with c2:
            c_k = st.number_input("k (tetangga)", step=1, min_value=1, key="c_k")
        with c3:
            c_w = st.selectbox("weighting", ["similarity", "uniform"], key="c_w")
        note = st.text_input("Catatan / alasan", key="c_note",
                             placeholder="mis. 'head RM-b sudah kuat → coba alpha lebih kecil'")
        eval_test = st.checkbox("Evaluasi TEST juga", key="c_test")
        config = {"alpha": c_alpha, "k": int(c_k), "weighting": c_w}

    if st.button(f"▶️ Jalankan {scenario.upper()} (1 konfigurasi)", type="primary",
                 disabled=alive, key="btn_run"):
        start_job({"phase": scenario, "out_dir": st.session_state["out_dir"], "smoke": smoke,
                   "note": note, "eval_test": eval_test, "config": config})

    st.divider()
    with st.expander("🧮 Mode Batch — jalankan banyak konfigurasi sekaligus", expanded=False):
        st.caption("Aditif terhadap mode di atas — form single-config tetap berfungsi seperti biasa. "
                   "Cocok untuk menjalankan rancangan grid seperti di folder tuning_grids/ dalam satu aksi.")
        batch_kind = st.radio("Cara membuat batch", ["Grid nilai (Cartesian)", "Tempel/unggah tabel CSV"],
                              key=f"batch_kind_{scenario}", horizontal=True)
        field_names = [n for n, _ in FIELD_SPECS[scenario]]

        if batch_kind == "Grid nilai (Cartesian)":
            st.caption("Isi tiap field dengan 1 nilai (tetap) atau beberapa nilai dipisah koma "
                       "(jadi sumbu grid). Contoh `lr`: `1e-5,2e-5,3e-5,5e-5`.")
            parsed, bad_fields = {}, []
            cols = st.columns(2)
            for i, (fname, ftype) in enumerate(FIELD_SPECS[scenario]):
                cur = st.session_state.get(SCEN_FIELD_KEY[scenario].get(fname), "")
                txt = cols[i % 2].text_input(fname, value=str(cur), key=f"grid_{scenario}_{fname}")
                vals, seen = [], set()
                for raw in txt.split(","):
                    raw = raw.strip()
                    if not raw or raw in seen:
                        continue
                    try:
                        v = ftype(raw)
                    except ValueError:
                        bad_fields.append(fname); continue
                    seen.add(raw); vals.append(v)
                parsed[fname] = vals

            if bad_fields:
                st.error(f"Nilai tidak valid pada field: {sorted(set(bad_fields))}")
            elif any(len(v) == 0 for v in parsed.values()):
                st.warning("Semua field harus punya minimal 1 nilai.")
            else:
                axis_fields = [f for f in field_names if len(parsed[f]) > 1]
                locked_fields = [f for f in field_names if len(parsed[f]) == 1]
                common_reason = st.text_area("Alasan umum grid (opsional, dipakai di semua baris)",
                                             key=f"grid_reason_{scenario}",
                                             placeholder="mis. 'memetakan interaksi lr×epochs×batch'")
                combos = list(itertools.product(*(parsed[f] for f in field_names)))
                preview_rows = []
                for combo in combos:
                    cfg = dict(zip(field_names, combo))
                    preview_rows.append({**cfg,
                                         "catatan": make_batch_note(scenario, cfg, axis_fields,
                                                                    locked_fields, common_reason),
                                         "eval_test": False})
                n = len(preview_rows)
                if n < 2:
                    st.info("Minimal 2 field harus bervariasi untuk membentuk grid (>=1 kombinasi "
                            "lebih dari 1) — kalau hanya 1 konfigurasi, pakai form single-config di atas.")
                else:
                    hist_p = out_dir() / f"runs_{scenario}.csv"
                    tcol = BATCH_TIME_COL[scenario]
                    if hist_p.exists():
                        hd = pd.read_csv(hist_p)
                        avg_t = hd[tcol].mean() if tcol in hd.columns and hd[tcol].notna().any() else None
                    else:
                        avg_t = None
                    src_label = "rata-rata histori" if avg_t else "estimasi kasar, belum ada histori"
                    avg_t = avg_t or BATCH_TIME_DEFAULT_S[scenario]
                    st.caption(f"**{n} konfigurasi** ≈ {n * avg_t / 60:.1f} menit "
                              f"({src_label}: {avg_t:.1f} dtk/konfigurasi)")
                    edited = st.data_editor(pd.DataFrame(preview_rows), width='stretch', height=280,
                                            key=f"grid_editor_{scenario}", num_rows="fixed")
                    if st.button(f"▶️ Jalankan batch grid ({n} konfigurasi)", type="primary",
                                disabled=alive, key=f"btn_grid_run_{scenario}"):
                        configs_payload = []
                        for _, r in edited.iterrows():
                            cfg = {}
                            for fname, ftype in FIELD_SPECS[scenario]:
                                cfg[fname] = ftype(r[fname]) if ftype is not str else str(r[fname])
                            configs_payload.append({"config": cfg, "note": str(r.get("catatan", "")),
                                                    "eval_test": bool(r.get("eval_test", False))})
                        bid = f"{scenario}_grid_{time.strftime('%Y%m%d_%H%M%S')}"
                        start_job({"phase": scenario, "out_dir": st.session_state["out_dir"],
                                   "smoke": smoke, "batch_id": bid, "configs": configs_payload})

        else:  # Tempel/unggah tabel CSV
            st.caption("Kolom = nama field HP (+ opsional `catatan`, `eval_test`). Kolom yang tidak "
                       "diisi memakai nilai default skenario. Cocok untuk baris non-grid (mis. "
                       "tahap coordinate-descent) atau mengunggah CSV dari folder tuning_grids/.")
            if st.button("📋 Isi dari config terbaik saat ini", key=f"btn_best_csv_{scenario}"):
                if scenario in best:
                    bc = best[scenario]["config"]
                    header = ",".join(list(bc.keys()) + ["catatan"])
                    vals = ",".join(str(v) for v in bc.values()) + ",turunan dari config terbaik saat ini"
                    st.session_state[f"csv_paste_{scenario}"] = header + "\n" + vals
                else:
                    st.warning("Belum ada config terbaik untuk skenario ini.")
            up = st.file_uploader("Upload CSV", type=["csv"], key=f"csv_upload_{scenario}")
            pasted = st.text_area("...atau tempel tabel CSV di sini", key=f"csv_paste_{scenario}",
                                  height=120)
            csv_df = None
            try:
                if up is not None:
                    csv_df = pd.read_csv(up)
                elif pasted.strip():
                    csv_df = pd.read_csv(io.StringIO(pasted))
            except Exception as e:
                st.error(f"Gagal parse CSV: {e}")
            if csv_df is not None:
                known_cols = {n for n in field_names} | {"catatan", "eval_test"}
                bad_cols = [c for c in csv_df.columns if c not in known_cols]
                if bad_cols:
                    st.error(f"Kolom tak dikenal: {bad_cols}. Kolom valid: {sorted(known_cols)}")
                else:
                    n_csv = len(csv_df)
                    st.dataframe(csv_df, width='stretch', height=200)
                    if st.button(f"▶️ Jalankan batch CSV ({n_csv} konfigurasi)", type="primary",
                                disabled=alive or n_csv < 1, key=f"btn_csv_run_{scenario}"):
                        configs_payload = []
                        for _, r in csv_df.iterrows():
                            cfg = {}
                            for fname, ftype in FIELD_SPECS[scenario]:
                                if fname in csv_df.columns and pd.notna(r[fname]):
                                    cfg[fname] = ftype(r[fname]) if ftype is not str else str(r[fname])
                            note_v = str(r["catatan"]) if "catatan" in csv_df.columns and pd.notna(r.get("catatan")) else ""
                            et_v = bool(r["eval_test"]) if "eval_test" in csv_df.columns and pd.notna(r.get("eval_test")) else False
                            configs_payload.append({"config": cfg, "note": note_v, "eval_test": et_v})
                        bid = f"{scenario}_csv_{time.strftime('%Y%m%d_%H%M%S')}"
                        start_job({"phase": scenario, "out_dir": st.session_state["out_dir"],
                                   "smoke": smoke, "batch_id": bid, "configs": configs_payload})

    if st.button("🔄 Regenerate figur & ringkasan grid", key=f"btn_regen_{scenario}"):
        import reporting
        written = reporting.grid_pivot_and_heatmap(out_dir(), scenario)
        reporting.tradeoff_scatter(out_dir(), scenario)
        reporting.top_configs_bar_chart(out_dir(), scenario)
        reporting.write_run_bundle_summary(out_dir())
        st.success(f"Dibuat ulang: {len(written)} file pivot/heatmap + top-config + tradeoff + "
                  f"tuning_summary.json")

    st.divider()
    show_runs(scenario, RUN_COLS[scenario])

    fdir = out_dir() / "figures"
    if fdir.exists():
        champ_curve = champ_conf = champ_pr = None
        if scenario in best:
            brid = best[scenario].get("run_id")
            cp = fdir / f"{scenario}_run{brid}_curve.png"
            champ_curve = cp if cp.exists() else None
        cc = fdir / f"{scenario}_best_val_confusion.png"
        champ_conf = cc if cc.exists() else None
        cprc = fdir / f"{scenario}_best_val_pr.png"
        champ_pr = cprc if cprc.exists() else None
        topbar = fdir / f"{scenario}_top_configs.png"
        tradeoff = fdir / f"{scenario}_tradeoff.png"
        heatmaps = sorted(fdir.glob(f"{scenario}_grid_heatmap_*.png"))
        if champ_curve or champ_conf or champ_pr or topbar.exists() or tradeoff.exists() or heatmaps:
            st.markdown("**📈 Figur**")
            c1, c2, c3 = st.columns(3)
            if champ_curve:
                c1.image(str(champ_curve), caption="Kurva training (juara saat ini)", width='stretch')
            if champ_conf:
                c2.image(str(champ_conf), caption="Confusion matrix VAL (juara saat ini)", width='stretch')
            if champ_pr:
                c3.image(str(champ_pr), caption="PR curve VAL (juara saat ini)", width='stretch')
            if topbar.exists():
                st.image(str(topbar), caption=f"Top konfigurasi ({scenario.upper()})", width='stretch')
            if tradeoff.exists():
                st.image(str(tradeoff), caption="Trade-off performa vs efisiensi", width='stretch')
            for hm in heatmaps:
                st.image(str(hm), caption=hm.stem, width='stretch')

# ----------------------------- 3 Final -----------------------------
with t_f:
    st.header("Final — Benchmark & Kriteria Sukses")
    st.markdown("Memakai **config terbaik sejauh ini** tiap skenario → evaluasi TEST + ukur "
                "**latency & peak memory inferensi RM-a/b/c dalam SATU sesi GPU yang sama** "
                "(angka efisiensi yang sah untuk Bab 4) + verdict kriteria sukses.")
    missing = [k for k in ["rma", "rmb", "rmc"] if k not in best]
    if missing:
        st.warning(f"Belum lengkap: {', '.join(x.upper() for x in missing)}. Jalankan dulu.")
    if st.button("▶️ Jalankan Final/Benchmark", type="primary", disabled=alive or bool(missing),
                 key="btn_f"):
        start_job({"phase": "final", "out_dir": st.session_state["out_dir"], "smoke": smoke})
    for f, cap in [("metrics/final_comparison.csv", "Perbandingan final (test)"),
                   ("metrics/inference_benchmark.csv", "Benchmark inferensi (satu sesi)"),
                   ("metrics/success_criteria.csv", "Kriteria sukses")]:
        p = out_dir() / f
        if p.exists():
            st.markdown(f"**{cap}**")
            st.dataframe(pd.read_csv(p), width='stretch')

    ffdir = out_dir() / "figures"
    if any((ffdir / f"final_{m}_confusion.png").exists() for m in ["rma", "rmb", "rmc"]):
        st.markdown("**Confusion matrix & PR curve (test, satu sesi)**")
        for m in ["rma", "rmb", "rmc"]:
            cconf, cpr = ffdir / f"final_{m}_confusion.png", ffdir / f"final_{m}_pr.png"
            if cconf.exists() or cpr.exists():
                fc1, fc2 = st.columns(2)
                if cconf.exists():
                    fc1.image(str(cconf), caption=f"{m.upper()} confusion (test)", width='stretch')
                if cpr.exists():
                    fc2.image(str(cpr), caption=f"{m.upper()} PR curve (test)", width='stretch')
    ib_p = ffdir / "final_inference_benchmark.png"
    if ib_p.exists():
        st.image(str(ib_p), caption="Latency & peak GPU memory inferensi (satu sesi)", width='stretch')

# ----------------------------- 4 Hasil -----------------------------
with t_r:
    st.header("Hasil & Unduh")
    o = out_dir()
    st.subheader("📋 Ringkasan siap copy-paste (kirim ke Claude untuk dianalisis)")
    L = ["# Hasil Run IndoBERT-with-RAC", ""]
    hw = read_json(o / "hardware.json")
    if hw:
        L += [f"Hardware: {hw.get('gpu')} | torch {hw.get('torch')} | CUDA {hw.get('cuda_version')}", ""]
    for sc in ["rma", "rmb", "rmc"]:
        p = o / f"runs_{sc}.csv"
        if p.exists():
            d = pd.read_csv(p)
            L += [f"## {sc.upper()} — {len(d)} run", "```", d.to_string(index=False)[:5000], "```", ""]
    for f, t in [("metrics/final_comparison.csv", "Perbandingan final"),
                 ("metrics/inference_benchmark.csv", "Benchmark inferensi (satu sesi)"),
                 ("metrics/success_criteria.csv", "Kriteria sukses")]:
        if (o / f).exists():
            L += [f"## {t}", "```", pd.read_csv(o / f).to_string(index=False), "```", ""]
    st.code("\n".join(L), language="markdown")

    st.subheader("📦 Unduh")
    light = [p for p in o.rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".png", ".log"}
             and "features" not in p.parts and "checkpoints" not in p.parts]
    c1, c2 = st.columns(2)
    with c1:
        if light:
            st.download_button("⬇️ Report bundle (ringan — kirim ke Claude)", make_zip(light, o),
                               file_name="report_bundle.zip", mime="application/zip", key="dl_rep")
            st.caption(f"{len(light)} file · riwayat run + catatan, metrik, figur, log.")
    with c2:
        if st.button("🗜️ Buat zip LENGKAP (+checkpoint & fitur)", key="btn_zip"):
            zp = ROOT / "results_full.zip"
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
                for p in o.rglob("*"):
                    if p.is_file():
                        z.write(p, p.relative_to(o.parent))
            st.success(f"`{zp}` ({zp.stat().st_size/1024**2:.0f} MB) — unduh via file browser / scp.")

    figs = sorted((o / "figures").glob("*.png")) if (o / "figures").exists() else []
    if figs:
        st.subheader("🖼️ Figur")
        groups = {
            "Confusion matrix (juara & final)": [f for f in figs if "confusion" in f.name],
            "Precision-Recall kelas judi": [f for f in figs if f.stem.endswith("_pr")],
            "Kurva training per run": [f for f in figs if f.stem.endswith("_curve")],
            "Heatmap grid": [f for f in figs if "grid_heatmap" in f.name],
            "Trade-off & top-config": [f for f in figs if "tradeoff" in f.name or "top_configs" in f.name],
        }
        used = {f for lst in groups.values() for f in lst}
        groups["Lainnya"] = [f for f in figs if f not in used]
        for label, flist in groups.items():
            if not flist:
                continue
            with st.expander(f"{label} ({len(flist)})", expanded=(label != "Kurva training per run")):
                cols = st.columns(3)
                for i, f in enumerate(flist):
                    cols[i % 3].image(str(f), caption=f.name, width='stretch')

if auto and alive:
    time.sleep(5)
    st.rerun()
