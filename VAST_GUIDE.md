# Panduan Vast.ai — Training & Tuning IndoBERT-with-RAC

Panduan operasional menjalankan **Tuning Control Panel** (Streamlit) di Vast.ai.
Estimasi biaya total: **~$0,15–0,40** (RTX 3090, ~30–60 mnt). Kredit $2 sangat cukup.

---

## 0. Ringkas (TL;DR)

1. Sewa **RTX 3090**, image torch **≥2.4**, **buka port 8501**, disk 30 GB.
2. Upload `dataset/`, `src/`, `app.py`.
3. `pip install streamlit transformers faiss-cpu scikit-learn pandas matplotlib`
4. `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
5. Buka `http://<IP>:<PORT-EKSTERNAL>` → atur HP → Run → pantau.
6. Tab **4 Hasil** → salin ringkasan / unduh `report_bundle.zip` → kirim ke Claude.
7. **DESTROY instance** (bukan Stop!).

---

## 1. Memilih instance (kriteria yang benar-benar menentukan)

| Kriteria | Nilai | Alasan |
|---|---|---|
| **GPU** | **RTX 3090 (24 GB)** | Kebutuhan nyata hanya ~3 GB VRAM (RM-a bs32 native). 3090 = paling banyak tersedia, murah (~$0,15–0,30/jam), cepat (~2,2× T4). |
| **Reliability** | **≥ 99%** | ⚠️ Risiko terbesar bukan tarif, tapi host mati di tengah run → kredit & waktu hangus. |
| **Tipe sewa** | **On-demand** (BUKAN interruptible) | Interruptible bisa di-preempt; hemat sedikit, risiko besar. |
| **Internet** | **≥ 100 Mbps** | Unduh bobot p2 (440 MB) + pip (~500 MB) **dibayar sebagai waktu**. |
| **Disk** | **30 GB** | Image runtime ~10–15 GB + model + hasil. Jangan minta 100 GB (disk ditagih). |
| **Image** | PyTorch **≥ 2.4**, CUDA 12.x, varian **runtime** | Kode dites di torch 2.12; `runtime` lebih kecil → pull lebih cepat (= lebih murah). |

> **Kenapa bukan A100/H100?** Model kita hanya 110M param — GPU besar tidak tersaturasi, tapi tarifnya bisa >$1,5/jam. Buang-buang kredit.

### ⚠️ WAJIB: buka port SEBELUM menyewa
Streamlit tidak punya tunnel bawaan. Saat membuat instance, di **Docker options / Open Ports** tambahkan:

```
-p 8501:8501
```

**Ini tidak bisa ditambahkan setelah instance dibuat** — kalau lupa, harus buat instance baru (kredit terbuang). Setelah jalan, Vast.ai menampilkan port eksternal (mis. `12345`) → akses `http://<IP>:12345`.

**Plan B (kalau port bermasalah / lupa):**
```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
cloudflared tunnel --url http://localhost:8501
```
→ dapat URL `https://xxx.trycloudflare.com` (tanpa akun, bisa dibuka dari HP).

---

## 2. Upload file (~2 MB)

Yang perlu diunggah (via Jupyter file browser Vast.ai, `scp`, atau git):

```
dataset/splits/{train,val,test}.csv
dataset/processed/metadata.json
src/*.py                 (preprocessing, dataset, modeling, evaluate, rac, tuning, job_runner, reporting)
app.py
tuning_grids/             (opsional — semua RMA/RMB/RMC_TUNING_GRID*.csv, hanya kalau mau
                            langsung pakai mode Batch tanpa mengetik ulang rancangan grid)
```

Bobot IndoBERT p2 (440 MB) **diunduh otomatis** dari HuggingFace saat job pertama.

Contoh scp (dari laptop):
```bash
scp -P <SSH_PORT> -r dataset src app.py tuning_grids root@<IP>:/workspace/IndoBERT-with-RAC/
```

---

## 3. Setup & jalankan (di terminal instance)

```bash
cd /workspace/IndoBERT-with-RAC
pip install -q streamlit transformers faiss-cpu scikit-learn pandas matplotlib
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Buka `http://<IP>:<PORT-EKSTERNAL>` dari browser/HP.

> **Job tetap jalan walau tab ditutup.** Training dijalankan sebagai proses terpisah
> (`src/job_runner.py`), bukan di dalam Streamlit. Tutup tab / HP mati / ganti jaringan
> → job aman; buka lagi URL-nya untuk lihat progres.

---

## 4. Alur kerja di UI — **1 klik = 1 konfigurasi**

Tidak ada pencarian otomatis. Strategi eksplorasi ada di tangan Anda:

```
set nilai HP + tulis alasan  →  Run (1 konfigurasi)  →  hasil val muncul
        ↑                                                     ↓
        └────── ubah nilai ←── analisis bareng Claude ────────┘
```

| Tab | Aksi | Waktu/run (3090) |
|---|---|---|
| **1 Status** | Cek GPU & data terbaca, lihat log | — |
| **2 Tuning** | Pilih **skenario** dari dropdown (RM-a / RM-b / RM-c), isi HP + **catatan alasan** → Run. Riwayat run di bawah mengikuti skenario yang dipilih. | lihat di bawah |
| **3 Final** | Run → evaluasi TEST + benchmark inferensi RM-a/b/c **satu sesi** + verdict kriteria | ~2 mnt |
| **4 Hasil** | Salin ringkasan → kirim ke Claude / unduh bundle | — |

Isi tab **Tuning** menyesuaikan skenario yang dipilih:

| Skenario | Field | Waktu/run (3090) |
|---|---|---|
| **RM-a** | lr / epochs / batch / warmup_ratio / weight_decay / micro_batch | ~2–3 mnt |
| **RM-b** | head_arch / hidden_dim / epochs / batch / lr / dropout / weight_decay. Fitur beku diekstrak otomatis (sekali, ~30 dtk) | ~detik |
| **RM-c** | α / k / weighting. Otomatis pakai **head RM-b terbaik** | ~detik |

> Nilai HP yang sudah diisi **tetap tersimpan** saat berpindah skenario — pindah ke RM-b lalu
> kembali ke RM-a tidak akan mereset field RM-a.

**Setiap run menumpuk** di `runs_{rma,rmb,rmc}.csv` beserta kolom `catatan`, plus grafik
tren val F1 antar-run. Config **terbaik-sejauh-ini** dilacak otomatis (`best.json`) dan
checkpoint-nya disimpan — dipakai tab Final.

> ⚠️ **Centang "Evaluasi TEST juga" seperlunya saja.** Pilih hyperparameter dari **val**;
> test idealnya dilihat di akhir (tab Final) agar tidak bias.

💡 **Coba dulu dengan centang "Mode SMOKE"** (subset kecil, ~10 dtk/run) untuk memastikan
semuanya jalan di instance sebelum run penuh.

### 🧮 Mode Batch — banyak konfigurasi dalam satu aksi

Aditif terhadap alur "1 klik = 1 konfigurasi" di atas (yang tetap berfungsi seperti biasa).
Di tab **Tuning**, buka expander **"🧮 Mode Batch — jalankan banyak konfigurasi sekaligus"**
di bawah tombol Run:

- **Grid nilai (Cartesian)** — isi tiap field dengan beberapa nilai dipisah koma (mis.
  `lr`: `1e-5,2e-5,3e-5,5e-5`) → otomatis jadi sumbu grid, `catatan` tiap baris terisi
  otomatis (bisa diedit di tabel pratinjau sebelum Run).
- **Tempel/unggah tabel CSV** — untuk rancangan yang sudah jadi tabel (kolom = nama field
  HP + `catatan`). **Unggah `tuning_grids/RMA_TUNING_GRID.csv`** untuk langsung menjalankan
  ke-24 konfigurasi Tahap 1 RM-a dalam satu klik, tanpa mengetik ulang (rancangan lengkap
  RM-a/b/c beserta setiap tahapnya: `tuning_grids/RMA_TUNING_GRID.md`,
  `RMB_TUNING_GRID.md`, `RMC_TUNING_GRID.md`). Tombol "Isi dari config terbaik saat ini"
  membantu menulis baris tahap lanjutan (coordinate descent di sel pemenang) tanpa
  mengetik ulang config yang sudah dikunci.

Batch berjalan sebagai **satu job** (tetap satu subprocess, tetap jalan meski tab
ditutup) — tiap config tetap menambah 1 baris ke `runs_{scenario}.csv` seperti biasa,
jadi progresnya terlihat live di tabel riwayat run. Jika satu config gagal (mis. OOM),
batch **tidak berhenti** — config itu dicatat ke `runs_{scenario}_errors.csv` dan lanjut
ke config berikutnya. Sidebar menampilkan progress bar + ETA saat batch berjalan, dengan
dua opsi stop: **"⏸ Hentikan setelah run ini"** (graceful, tunggu config saat ini selesai)
atau **"⛔ Hentikan paksa"** (SIGTERM langsung, seperti stop biasa).

**Untuk SETIAP config** (single atau batch), **kurva training** dibuat otomatis begitu
run selesai — murah, tanpa GPU tambahan
(`figures/{scenario}_run{id}_curve.png` — loss & F1 per epoch, RM-a/RM-b saja karena
RM-c tak punya epoch). Kalau config itu jadi **juara-sejauh-ini**, ditambah 2 figur lagi:
**confusion matrix VAL** (`{scenario}_best_val_confusion.png`) dan **PR curve kelas judi**
(`{scenario}_best_val_pr.png`) — relevan karena imbalance kelas (~4,5:1) adalah tema
sentral skripsi ini.

Setelah batch selesai, artefak "grid sebagai permukaan" dibuat otomatis: pivot table
(`metrics/{scenario}_grid_pivot_*.csv`) + heatmap (`figures/{scenario}_grid_heatmap_*.png`)
per nilai `batch`/`hidden_dim`/`weighting`, scatter performa-vs-efisiensi
(`figures/{scenario}_tradeoff.png` — ukuran titik = `trainable_params`, **warna titik =
`peak_mem_mb`**), dan **bar chart top-10 konfigurasi** (`figures/{scenario}_top_configs.png`).
Tombol **"🔄 Regenerate figur & ringkasan grid"** di tab Tuning membuat ulang semua artefak
level-batch ini kapan saja (juga berguna untuk run single-config yang menumpuk jadi grid
dari waktu ke waktu).

**Metrik efisiensi per-run** kini lengkap di ketiga skenario: RM-a sudah punya
`train_time_s`/`peak_mem_mb`/`trainable_params` sejak awal; **RM-b kini juga merekam
`peak_mem_mb` dan `infer_latency_ms`** (latency 1-sampel, diukur murah lewat forward pass
head di atas fitur beku — tak butuh GPU tambahan) di kolom `runs_rmb.csv`, karena
`head_arch`/`hidden_dim` bisa mengubah biaya inferensi meski `trainable_params`-nya kecil.

**Figur langsung tampil di UI**, bukan cuma tersimpan di disk:
- Tab **2 Tuning** — di bawah tabel riwayat run: kurva training + confusion matrix + PR
  curve juara, bar chart top-config, heatmap grid, scatter trade-off (skenario yang sedang
  dipilih).
- Tab **3 Final** — confusion matrix + PR curve test untuk RM-a/RM-b/RM-c berdampingan,
  plus bar chart perbandingan latency & peak GPU memory inferensi ketiganya (satu sesi).
- Tab **4 Hasil** — semua PNG dikelompokkan ke expander per jenis (Confusion matrix /
  Precision-Recall / Kurva training per run / Heatmap grid / Trade-off & top-config /
  Lainnya) — grup "Kurva training per run" collapsed default karena bisa puluhan file.

### 💰 Catatan biaya untuk alur looping
Instance ditagih **selama menyala**, termasuk saat kita berdiskusi menganalisis hasil.
Di RTX 3090 (~$0,22/jam), kredit $2 ≈ **~9 jam** — cukup longgar untuk beberapa sesi
looping. Kalau mau jeda lama, unduh hasil dulu lalu **destroy**; nanti sewa lagi dan
upload kembali `results/vast/` untuk melanjutkan (riwayat run tetap menumpuk).

---

## 5. Mengambil hasil

Di tab **4 Hasil**:
1. **📋 Ringkasan copy-paste** — cara tercepat: salin, tempel ke chat Claude. Tanpa unduh.
2. **📦 Report bundle (~2–5 MB)** — semua CSV trial (termasuk `runs_{scenario}_errors.csv`
   bila ada config batch yang gagal) + kolom `catatan`, `tuning_summary.json` (ringkasan
   jumlah run/batch + juara-sejauh-ini per skenario), `hardware.json`, `best.json` (config
   PERSIS yang dipakai checkpoint juara tiap skenario — gantikan referensi lama
   `config_used.json`, yang tidak pernah benar-benar ditulis), dan seluruh figur: kurva
   training per run, confusion matrix + PR curve (juara & final), heatmap grid, scatter
   trade-off, bar chart top-config. **Cukup untuk analisis penuh.**
3. **🗜️ Zip lengkap** — plus checkpoint & fitur (~500 MB+). Hanya bila perlu model-nya.

---

## 6. ⚠️ Setelah selesai: **DESTROY**, bukan Stop

- **Stop** = instance mati tapi **disk tetap ditagih**.
- **Destroy** = berhenti total, tidak ada tagihan.

**Pastikan hasil sudah diunduh sebelum destroy** — data ikut terhapus.

---

## 7. Estimasi biaya

| Tahap | Waktu (RTX 3090) |
|---|---|
| Setup (pip + unduh bobot p2) | ~5–8 mnt |
| Tuning RM-a | ~25–35 mnt |
| Tuning RM-b | ~3 mnt |
| Tuning RM-c | ~1 mnt |
| Final + benchmark | ~3 mnt |
| **Total** | **~40–50 mnt ≈ $0,15–0,25** |

Sisa kredit aman untuk percobaan ulang bila perlu.

---

## 8. Troubleshooting

| Masalah | Solusi |
|---|---|
| UI tak bisa diakses | Port 8501 tidak dibuka saat sewa → pakai **cloudflared** (Plan B §1). |
| `CUDA out of memory` | Turunkan `micro_batch` di tab Tuning (skenario RM-a; 32 → 16). Di 3090 seharusnya tak terjadi. |
| Job mati/terputus | Hasil trial **tetap aman** (CSV ditulis inkremental). Jalankan ulang fase yang gagal — trial yang sudah ada tidak hilang. |
| UI error / Streamlit bermasalah | Jalankan manual: `python src/job_runner.py --config results/vast/job.json` |
| Bobot p2 lambat diunduh | Normal ~30 dtk di datacenter. Bila macet, ulangi (resume otomatis). |
