# HASIL — Berkas Rujukan Tunggal

Semua angka hasil training, benchmark, dan eksplorasi dikumpulkan di sini supaya tidak perlu
membuka banyak folder. **Kalau ragu: pakai yang ditandai ✅ RESMI.** Berkas ini menyalin isi
artefak, tidak menggantikannya — tidak ada berkas yang dipindah/dihapus untuk membuatnya.

Disusun 2026-08-16 dari isi repositori apa adanya. Pendamping: `AUDIT_PIPELINE.md` (verifikasi
kode vs klaim naskah), `PROGRESS.md` (status pekerjaan), `CLAUDE.md` (panduan teknis).

---

## 1. Jawaban singkat: data mana yang dipakai?

| Pertanyaan | Jawaban |
|---|---|
| Angka Bab 4 diambil dari mana? | **`results/vast/`** — satu-satunya sumber resmi |
| Berapa hasil finalnya? | §3 di bawah (RM-a 0,9607 · RM-b 0,9486 · RM-c 0,9497 F1-macro test) |
| Diukur di mana? | RTX 3090 (Vast.ai), **satu sesi**, 2026-07-25 08:02–10:05 |
| Data latih yang dipakai? | `dataset/splits/{train,val,test}.csv`, kolom **`text_clean`** |
| Folder hasil lain (`results_c2/`, `vast_rmc_cheap_head/`, `results/local/`, `results/features/`)? | **Bukan untuk Bab 4** — lihat §2 |
| Kenapa banyak folder mirip? | Sisa eksplorasi tambahan setelah kampanye resmi selesai; sengaja dipisah supaya `results/vast/` tidak tercemar |

**Satu kalimat untuk diingat:** angka performa (F1) boleh dibandingkan lintas folder; angka
efisiensi (waktu, memori, latency) **tidak boleh** — beda folder = beda mesin/sesi.

---

## 2. Peta folder hasil

| Folder | Isi | Status |
|---|---|---|
| `results/vast/` | Kampanye tuning penuh (124 run) + benchmark final test | ✅ **RESMI — sumber Bab 4** |
| `results/vast/metrics/` | `final_comparison.csv`, `inference_benchmark.csv`, `success_criteria.csv` + 9 pivot grid | ✅ **RESMI — tabel jadi** |
| `results/vast/figures/` | 83 PNG: kurva per-run, heatmap grid, confusion matrix, PR curve | ✅ RESMI — gambar Bab 4 |
| `results/vast/checkpoints/` | `rma_best.pt` (418 MB), `rmb_best.pt`, `rmc_best.pt` | ✅ RESMI — bobot final |
| `results/figures/` | 5 PNG EDA (distribusi label, panjang teks, temporal) | ✅ RESMI — untuk Bab 3/4.2 |
| `results/combined/` | 3 CSV gabungan lintas-folder (`source` + `run_id`) | 🔵 Turunan — untuk analisis lintas eksperimen saja |
| `results/faiss_index/` | Benchmark modul FAISS (build & search) | 🟡 Pendukung — **diukur di laptop CPU, bukan 3090** |
| `results/vast_rmc_cheap_head/` | Ablasi biaya: RM-c di atas head RM-b termurah (linear, 1.538 param) | 🟡 Eksplorasi tambahan (2026-08-03) |
| `results_c2/` | Kandidat #2 (config runner-up) untuk uji sensitivitas | 🟡 Eksplorasi tambahan |
| `results/features/` | Cache embedding lama, **beda isi** dari `results/vast/features/` | ⛔ Usang — jangan dipakai |
| `results/local/` | Output `run_local_training.py` | ⛔ **KOSONG** — tidak pernah dijalankan sejak clean-slate |

---

## 3. ✅ HASIL FINAL BAB 4 (`results/vast/`)

### 3.1 Lingkungan

| Item | Nilai |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24.124 MB VRAM |
| Software | torch 2.12.0+cu130, CUDA 13.0, transformers 5.14.1 |
| Base model | `indobenchmark/indobert-base-p2`, max_length 128 |
| Special token | `[URL]`, `[MENTION]`, `[NUM]` (+ `resize_token_embeddings`) |
| Tanggal | 2026-07-25, satu sesi (tuning 08:02 → benchmark final 10:05) |
| Seed | 42 (split maupun training) |

### 3.2 Data

| Tahap | Baris |
|---|---|
| Mentah (`dataset/raw/data_labeling.csv`) | 14.237 |
| Setelah buang kosong | 14.227 |
| Setelah dedup NFKC-exact | 9.412 |
| Setelah guard anti-leakage (buang 17 dari val/test) | **9.395** |

Split stratified 70:15:15 → **train 6.588 · val 1.402 · test 1.405** (~18,2% kelas judi, rasio 4,5:1).
Class weight dari train: **{0: 0,6110 · 1: 2,7519}**, dipakai di `CrossEntropyLoss` RM-a **dan** RM-b.

### 3.3 Konfigurasi final tiap skenario

| Model | Konfigurasi | Asal |
|---|---|---|
| **RM-a** | `lr=2e-5, epochs=5, batch=32, warmup_ratio=0.1, weight_decay=0.01` (best epoch 4) | run #13 dari 26 |
| **RM-b** | `head=mlp, hidden_dim=1024, lr=1e-3, epochs=10, dropout=0.1, weight_decay=0.0, batch=32` (best epoch 6) | run #28 dari 31 |
| **RM-c** | `alpha=0.2, k=5, weighting=similarity` + head RM-b di atas | sel α=0,2/k=5 (baris run #15) |

⚠️ `best.json` menulis juara RM-c sebagai `run_id: 13`, padahal konfigurasi itu ada di baris
**run #15**; run #13 justru `k=1`. **Rujuk konfigurasinya (α=0,2; k=5), jangan nomor run-nya.**
Lihat §6.

### 3.4 Performa pada TEST set (1.405 sampel)

| Model | F1-macro | Akurasi | F1 judi | Precision judi | Recall judi | val F1-macro |
|---|---|---|---|---|---|---|
| RM-a | **0,960653** | 0,976512 | 0,935673 | 0,933852 | 0,937500 | 0,982160 |
| RM-b | 0,948574 | 0,969395 | 0,915851 | 0,917647 | 0,914062 | 0,965301 |
| RM-c | 0,949693 | 0,970107 | 0,917647 | 0,921260 | 0,914062 | 0,969903 |

**Confusion matrix test** (direkonstruksi dari precision/recall/support; akurasinya cocok sampai
6 desimal dengan `final_comparison.csv`, dan bisa dicek silang ke `figures/final_*_confusion.png`):

| Model | TP (judi benar) | FN (judi lolos) | FP (salah tuduh) | TN |
|---|---|---|---|---|
| RM-a | 240 | 16 | 17 | 1.132 |
| RM-b | 234 | 22 | 21 | 1.128 |
| RM-c | 234 | 22 | **20** | 1.129 |

RM-c vs RM-b: selisihnya **tepat 1 false positive**. Recall identik. Jangan diklaim lebih dari itu.

### 3.5 Efisiensi (satu sesi RTX 3090)

| Metrik | RM-a | RM-b | RM-c |
|---|---|---|---|
| Trainable params | 109.485.314 | 789.506 (0,72%) | 0 |
| Waktu latih | 81,10 s | 11,65 s (7,90 ekstraksi + 3,75 head) | 0 s |
| Peak memory latih | 3.091,8 **MiB** | 63,9 **MiB** | — |
| Latency inferensi | 9,3978 ms | 9,0789 ms | 10,9568 ms |
| Peak memory inferensi | 1.473,6 MiB | 1.273,9 MiB | 1.273,9 MiB |

⚠️ Angka memori adalah **MiB yang selama ini ditulis MB** → 3.091,8 MiB = **3.242,0 MB**;
63,9 MiB = **67,0 MB**. Persentase penghematan tidak berubah. Lihat §6.

### 3.6 Kriteria sukses

| Model | Gap F1 | ≤3pp? | Reduksi param | ≥90%? | Reduksi waktu | ≥50%? | Total |
|---|---|---|---|---|---|---|---|
| RM-b | 1,21 pp | ✅ | 99,2789% | ✅ | 85,64% | ✅ | **3/3** |
| RM-c | 1,10 pp | ✅ | 100% | ✅ | 100% | ✅ | **3/3** |

Syarat kompetitif hanya 2/3 → **keduanya lolos dengan selisih lebar.**

---

## 4. Rekap tuning (dasar pemilihan config final)

Total **124 run**, semuanya dipilih **hanya dari validation** — tidak ada satu pun kolom `test_*`
di ketiga CSV run (terverifikasi di `AUDIT_PIPELINE.md` §C6).

| Skenario | Jumlah run | Batch | Waktu | Juara |
|---|---|---|---|---|
| RM-a | 26 | 2 (24 grid + 2 coordinate descent) | 08:02–09:06 | run #13 · val 0,982160 |
| RM-b | **31** | 5 (4+4+2+15+6) | 09:31–09:48 | run #28 · val 0,965301 |
| RM-c | 67 | 2 (66 grid α×k + 1 cek weighting) | 09:54–09:59 | α=0,2/k=5 · val 0,969903 |

⚠️ `PROGRESS.md` menulis RM-b "27 run"; jumlah baris sebenarnya di `runs_rmb.csv` dan di
`tuning_summary.json` adalah **31**. Pakai 31.

### 4.1 RM-a — grid `lr × epochs`, val F1-macro

**batch 16:**

| lr \ epochs | 3 | 5 | 8 |
|---|---|---|---|
| 1e-5 | 0,9681 | 0,9785 | 0,9774 |
| 2e-5 | 0,9773 | 0,9749 | 0,9763 |
| 3e-5 | 0,9704 | 0,9717 | 0,9711 |
| 5e-5 | 0,9670 | 0,9747 | 0,9762 |

**batch 32:**

| lr \ epochs | 3 | 5 | 8 |
|---|---|---|---|
| 1e-5 | 0,9749 | 0,9760 | 0,9796 |
| 2e-5 | 0,9735 | **0,9822** | 0,9736 |
| 3e-5 | 0,9796 | 0,9784 | 0,9773 |
| 5e-5 | 0,9713 | 0,9749 | 0,9735 |

Juara berada di **tengah grid** (bukan di tepi) → rentang pencarian sudah memadai.
Coordinate descent tahap 2: `warmup_ratio 0,1→0,0` dan `weight_decay 0,01→0,10` — run #26
(wd=0,10) menghasilkan val F1 **identik** 0,982160 dengan waktu 80,8 s; dipilih run #13 (wd=0,01)
sesuai default literatur.

**5 run terbaik RM-a:**

| run | lr | epochs | batch | wd | val F1-macro | val F1 judi | best epoch | waktu | peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| **13** | 2e-5 | 5 | 32 | 0,01 | **0,982160** | 0,970874 | 4 | 81,1 s | 3.091,8 |
| 26 | 2e-5 | 5 | 32 | 0,10 | 0,982160 | 0,970874 | 4 | 80,8 s | 3.112,1 |
| 18 | 1e-5 | 8 | 32 | 0,01 | 0,979597 | 0,966601 | 6 | 130,5 s | 3.098,3 |
| 20 | 3e-5 | 3 | 32 | 0,01 | 0,979597 | 0,966601 | 3 | 48,8 s | 3.098,3 |
| 4 | 1e-5 | 5 | 16 | 0,01 | 0,978495 | 0,964844 | 5 | 142,9 s | 2.320,5 |

### 4.2 RM-b — grid `lr × epochs` pada `hidden_dim=1024`, val F1-macro

| lr \ epochs | 5 | 10 | 20 | 30 |
|---|---|---|---|---|
| 1e-4 | 0,9413 | 0,9561 | 0,9612 | 0,9628 |
| 2e-4 | 0,9543 | 0,9618 | 0,9629 | 0,9637 |
| 5e-4 | 0,9579 | 0,9613 | 0,9640 | 0,9640 |
| 1e-3 | 0,9595 | **0,9653** | 0,9650 | 0,9650 |

Pembanding arsitektur: head **linear** hanya mencapai val 0,9125 (`hidden_dim` tak relevan),
head **MLP/128** 0,9368 → kapasitas head naik = performa naik sampai `hidden_dim=1024`.

**5 run terbaik RM-b:**

| run | arch | hidden | epochs | lr | wd | val F1-macro | val F1 judi | best epoch | waktu head |
|---|---|---|---|---|---|---|---|---|---|
| **28** | mlp | 1024 | 10 | 1e-3 | 0,00 | **0,965301** | 0,943249 | 6 | 3,75 s |
| 23 | mlp | 1024 | 10 | 1e-3 | 0,01 | 0,964980 | 0,942574 | 6 | 3,97 s |
| 24 | mlp | 1024 | 20 | 1e-3 | 0,01 | 0,964980 | 0,942574 | 6 | 7,91 s |
| 25 | mlp | 1024 | 30 | 1e-3 | 0,01 | 0,964980 | 0,942574 | 6 | 11,71 s |
| 20 | mlp | 1024 | 20 | 5e-4 | 0,01 | 0,964049 | 0,941176 | 12 | 7,31 s |

Empat run teratas praktis seri (≤0,13 pp) — dipilih yang termurah yang juga tertinggi.

### 4.3 RM-c — grid `alpha × k` (weighting=similarity), val F1-macro

| α \ k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| 0,0 | 0,9653 | 0,9653 | 0,9653 | 0,9653 | 0,9653 | 0,9653 |
| 0,1 | 0,9653 | 0,9628 | 0,9652 | 0,9652 | 0,9652 | 0,9639 |
| **0,2** | *0,9700* | 0,9687 | **0,9699** | 0,9662 | 0,9662 | 0,9650 |
| 0,3 | 0,9675 | 0,9673 | 0,9673 | 0,9661 | 0,9673 | 0,9661 |
| 0,4 | 0,9687 | 0,9686 | 0,9686 | 0,9685 | 0,9672 | 0,9672 |
| 0,5 | 0,9382 | 0,9658 | 0,9684 | 0,9658 | 0,9671 | 0,9633 |
| 0,6 | 0,9382 | 0,9658 | 0,9643 | 0,9594 | 0,9591 | 0,9580 |
| 0,7 | 0,9382 | 0,9607 | 0,9582 | 0,9553 | 0,9527 | 0,9472 |
| 0,8 | 0,9382 | 0,9410 | 0,9517 | 0,9526 | 0,9462 | 0,9370 |
| 0,9 | 0,9382 | 0,9397 | 0,9414 | 0,9460 | 0,9424 | 0,9219 |
| 1,0 | 0,9382 | 0,9397 | 0,9414 | 0,9437 | 0,9383 | 0,9075 |

- α=0,0 = RM-b murni (0,9653); α=1,0 = retrieval murni (turun tajam) → **fusi memang berguna,
  tapi hanya pada bobot kecil**.
- Sel tertinggi secara mekanis adalah α=0,2/**k=1** (0,969995), **tidak dipilih** karena k=1
  bergantung pada satu tetangga (rapuh); dipilih α=0,2/**k=5** (0,969903, selisih 0,009 pp = seri).
  Alasan ini tercatat di `catatan` run #67 dan `run_candidate2_vast.sh:76`.
- `weighting=uniform` diuji pada sel juara (run #67): hasilnya **sama persis** 0,969903 →
  skema pembobotan tidak berpengaruh di titik itu.

---

## 5. Angka pendukung — 🟡 BUKAN untuk tabel utama Bab 4

Boleh dikutip sebagai pembahasan/limitasi, **asal disebut mesinnya berbeda**.

### 5.1 Benchmark modul FAISS (`results/faiss_index/`) — laptop, CPU

| Item | Nilai |
|---|---|
| Indeks | `IndexFlatIP` (eksak), 6.588 × 768, float32, 19,3 MB |
| Komposisi label | 5.391 non-judi / 1.197 judi |
| Waktu bangun | 43,04 ms (± 3,47; normalisasi 28,5 + add 14,5) |
| Search k=5 | 0,0703 ms/query (test), 0,0779 ms/query (val) |
| Rata-rata similaritas top-1 | 0,8109 (test) / 0,8113 (val) |
| Mesin | AryaLaptop, Windows 11, Intel 16-thread, faiss 1.14.3, 2026-08-12 |

Berguna untuk menunjukkan biaya retrieval sangat kecil — tapi **jangan** dijejerkan dengan
latency 3090 di §3.5.

### 5.2 Ablasi head termurah (`results/vast_rmc_cheap_head/`, 2026-08-03)

Head RM-b linear (**1.538 param**, val 0,9125) + RAC → val naik ke **0,9576** (α=0,5, k=3),
test F1-macro **0,9513**. Artinya RAC menutup sebagian besar jurang head murah, tapi tetap di bawah
kombinasi resmi. 66 dari 69 run di folder ini val-only; 3 run terakhir membuka test.

### 5.3 Kandidat #2 / uji sensitivitas (`results_c2/`)

`results_c2/final_comparison_all_candidates.csv` membandingkan 8 baris (resmi vs runner-up):
RM-a wd=0,10 → test 0,960772 (vs 0,960653); RM-b wd=0,01 → test **0,954062** (lebih tinggi dari
config resmi 0,948574, tapi val-nya lebih rendah 0,964980 vs 0,965301 — **jangan ditukar**, itu
akan berarti memilih lewat test); RM-c α=0,2/k=3 → test 0,948574.

### 5.4 Latency di GPU lain (`results/vast_rmc_cheap_head/metrics_latency_local_rtx3050.csv`)

RTX 3050: RM-a 16,27 ms · RM-b 14,67 ms · RM-c 17,11 ms · RM-b cheap 13,52 ms · RM-c cheap 16,15 ms.
Pola urutannya sama dengan 3090; angkanya **tidak boleh dicampur**.

---

## 6. Koreksi yang harus dibawa ke naskah

Dari `AUDIT_PIPELINE.md` (verifikasi kode vs klaim). Empat hal yang mengubah tulisan, bukan angka:

| # | Hal | Yang benar |
|---|---|---|
| 1 | Representasi embedding | **Mean-pooling** bertopeng atas seluruh token, **bukan token `[CLS]`** (`src/modeling.py:78-87`). Berlaku untuk head RM-b maupun indeks RM-c. |
| 2 | Satuan memori | Angka adalah **MiB** (`/1024**2`). 3.091,8 MiB = 3.242,0 MB; 63,9 MiB = 67,0 MB. Persentase aman. Juga: 63,9 hanya fase head — puncak ekstraksi fitur 529,5 MiB tidak termasuk. |
| 3 | Penyaringan data | **Tidak ada** filter panjang minimum / deteksi bahasa / pembuangan non-teks. Hanya 3 tahap: kosong, dedup, guard. (Bab 3 perlu diselaraskan dengan Bab 4.) |
| 4 | Nomor run RM-c | `best.json` menulis `run_id 13` untuk config `k=5`, tapi di CSV baris itu adalah run #15. Rujuk **config**, jangan nomor run. |

Dua hal lagi soal cara menyebut, bukan salah:

- **Waktu latih 81,10 s vs 11,65 s tidak identik cakupannya.** 81,10 s memuat 5× evaluasi validasi
  ber-encoder penuh + penyalinan 109,5 juta parameter; 11,65 s memuat ekstraksi **tiga** split
  (termasuk test) yang diukur sekali lalu dipakai ulang di 31 run. Angka 85,64% tetap boleh
  dikutip, asal cakupannya dinyatakan.
- **Latency RM-c (+1,88 ms) bukan murni biaya retrieval** — jalur RM-c menanggung sinkronisasi
  GPU→CPU tiap iterasi yang tidak dialami RM-a/RM-b.

---

## 7. Inventaris berkas

### `results/vast/` — ✅ resmi

| Berkas | Isi |
|---|---|
| `runs_rma.csv` / `runs_rmb.csv` / `runs_rmc.csv` | 26 / 31 / 67 baris; config + metrik val + kolom `catatan` (alasan tiap run) |
| `best.json` | Config juara per skenario (⚠️ `run_id` RM-c geser, §6) |
| `tuning_summary.json` | Jumlah run, rentang waktu, juara per skenario |
| `hardware.json` | GPU/CUDA/torch sesi pengukuran |
| `metrics/final_comparison.csv` | **Tabel utama Bab 4** (test + efisiensi, 3 baris) |
| `metrics/inference_benchmark.csv` | Latency + peak memory inferensi, satu sesi |
| `metrics/success_criteria.csv` | Verdict 3 kriteria untuk RM-b & RM-c |
| `metrics/*_grid_pivot_*.csv` | 9 pivot grid (sudah disalin ke §4) |
| `history/rma_history.csv`, `rmb_history.csv` | Kurva val per epoch, kolom `run_id` (untuk deteksi overfitting) |
| `history/_backup_per_run/` | 57 file format lama; sudah digabung, disimpan sebagai cadangan |
| `figures/` | 83 PNG (26+31 kurva run, 9 heatmap, confusion & PR final, tradeoff) |
| `checkpoints/rma_best.pt` (418 MB), `rmb_best.pt`, `rmc_best.pt` | Bobot & config final |
| `features/indobenchmark__indobert-base-p2/` | Cache embedding beku 6.588/1.402/1.405 × 768 + `extract_meta.json` (7,9 s, 529,5 MiB) |
| `job.json`, `job.log`, `job.pid`, `progress.json` | Jejak job terakhir (fase `final`) — log memuat cetakan tabel final |

### Folder lain

| Berkas | Isi | Status |
|---|---|---|
| `results/combined/runs_{rma,rmb,rmc}.csv` (+ `README.md`) | 27 / 34 / 137 baris gabungan, kolom `source` | 🔵 turunan `scripts/merge_runs.py` |
| `results/faiss_index/{faiss_index_build,faiss_search_latency}.csv`, `train_index.faiss` | Benchmark FAISS (laptop CPU) | 🟡 §5.1 |
| `results/vast_rmc_cheap_head/` | `runs_rmb_cheap.csv` (1), `runs_rmc_cheap.csv` (69), `best.json`, `metrics_latency_local_rtx3050.csv`, 12 figur | 🟡 §5.2, §5.4 |
| `results_c2/final_comparison_all_candidates.csv` | 8 baris perbandingan resmi vs kandidat #2 | 🟡 §5.3 |
| `results_c2/vast_candidate2/` | `runs_{rma,rmb,rmc}_c2.csv` (1/2/1 baris, **ada kolom test**), `best_c2.json`, checkpoint | 🟡 §5.3 |
| `results/figures/` (5 PNG) | Figur EDA | ✅ Bab 3/4.2 |
| `results/features/` | Cache embedding **lama** (md5 beda dari `results/vast/features/`) | ⛔ usang |
| `results/local/` | 5 subfolder kosong | ⛔ kosong |
| `draft_bab4/draf_bab_4-1..8.md` | Draf naskah §4.1–4.8 + `data_mentah_bab4.csv` | 📝 tulisan |
| `tuning_grids/*.md` + `*.csv` | Rancangan grid & alasan tiap sel (3 skenario, 8 CSV tahap) | 📋 rencana yang dieksekusi |

### Berkas kembar (belum saya sentuh)

Empat salinan cache embedding 28 MB dan dua checkpoint RM-a 418 MB:

| Path | Ukuran | Catatan |
|---|---|---|
| `results/vast/features/…` | 28 MB | ✅ dipakai |
| `results_c2/vast_candidate2/features/…` | 28 MB | **identik** (md5 sama) dengan yang resmi |
| `results/vast_rmc_cheap_head/features/…` | 28 MB | isi sedikit berbeda (ekstraksi ulang sesi lain) |
| `results/features/` | 28 MB | ⛔ usang, beda isi |
| `results/vast/checkpoints/rma_best.pt` | 418 MB | ✅ dipakai |
| `results_c2/vast_candidate2/checkpoints/rma_best.pt` | 418 MB | kandidat #2 (isi beda) |

Total ~1 GB. Semuanya `.gitignore`-d dan bisa dibuat ulang. **Tidak ada yang saya hapus** — bilang
kalau mau dibersihkan.

---

## 8. Cara membuat ulang tiap artefak

| Artefak | Perintah |
|---|---|
| `dataset/splits/` + `metadata.json` | jalankan `notebooks/02_preprocessing.ipynb` (⚠️ menimpa split yang ada) |
| Run tuning baru | `streamlit run app.py` → tab Tuning (1 klik = 1 config) — **jangan** ke `results/vast/` |
| Benchmark final | tab Final di `app.py` (butuh `best.json` + checkpoint lengkap) |
| CSV gabungan | `python scripts/merge_runs.py` |
| Benchmark FAISS | `python scripts/benchmark_faiss_index.py` |
| Reproduksi lokal 3 skenario | `python run_local_training.py` → `results/local/` |

**Aturan yang berlaku terus:** eksperimen baru **wajib** memakai `out_dir` lain — `results/vast/`
memegang angka Bab 4 yang sudah terkunci.
