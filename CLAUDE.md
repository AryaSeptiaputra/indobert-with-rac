# CLAUDE.md

Panduan untuk Claude Code saat bekerja di repositori ini.

## Gaya kode: fungsi hidup di notebook

Repo ini mengikuti gaya `indonesian-rag-retrieval-benchmark`, bukan struktur
package `writer-code`. Invoke skill `writer-code` untuk aturan penulisan fungsi
(peran dan awalan nama, docstring Google style, komentar wajib di regex dan format
waktu, type hint, tanpa basa-basi), tetapi JANGAN membuat `src/`, `app/`,
`tests/`, `config.py`, atau `.env`: pengecualian §2.11 dan Bagian 1 disengaja.

- Sel kode pertama tiap notebook berisi seluruh konstanta (path relatif `../`,
  nama model, seed, ambang), masing-masing dengan komentar satu baris.
- Setiap fungsi ditulis di sel yang memakainya dan langsung dipanggil di
  bawahnya. Keluaran untuk pembaca memakai `print`/`display`, bukan logging.
- Tidak ada import antar-notebook. Fungsi yang dipakai beberapa notebook disalin.
  Saat mengubah salah satunya, ubah juga salinannya:

  | Fungsi | Ada di |
  |---|---|
  | `normalize_nfkc`, `clean_text` | 01, 02 |
  | `load_tokenizer` | 01, 02, 03a, 03b, 05 |
  | `to_tensor_dataset`, `init_special_token_embeddings` | 03a, 03b, 05 |
  | `build_finetune_model` | 03a, 05 |
  | `build_encoder`, `mean_pool` | 03b, 05 |
  | `build_head` | 03b, 03c, 05 |
  | `compute_metrics` | 03a, 03b, 03c, 05, 06 |
  | `load_json`, `save_json`, `load_runs`, `append_run`, `update_best` | 03a, 03b, 03c (sebagian di 05-07) |
  | `softmax`, `l2_normalize`, `build_index`, `to_retrieval_distribution`, `predict_rac` | 03c, 05 (`l2_normalize` juga di 07) |
  | `load_head`, `compute_head_logits`, `load_features` | 03c, 05 (`load_features` juga di 07) |
  | `record_tuning_session` / `verify_final_session` | 03a / 05 |
  | `to_rmc_grid`, `find_shared_config`, `summarize_rac_per_head`, `rank_runs`, `rank_rmc_runs` | 03c, 06 |
  | `compute_content_hash` | 03c, 05, 06 |

- Notebook terhubung hanya lewat berkas di `data/` dan `outputs/`. Format berkas
  itu (kolom CSV, kunci JSON, isi checkpoint) adalah kontrak antar-notebook;
  mengubahnya berarti mengubah seluruh pembacanya.

## Gambaran proyek

Skripsi S1 (ITENAS 2026) yang membandingkan tiga strategi adaptasi IndoBERT untuk
mendeteksi komentar promosi judi daring berbahasa Indonesia di YouTube.
Pertanyaan pusatnya adalah trade-off antara performa dan efisiensi:

| Kode | Strategi | Deskripsi |
|------|----------|-----------|
| RM-a | Full Fine-tuning | Seluruh parameter IndoBERT diperbarui (baseline) |
| RM-b | Frozen Encoder | Hanya classification head yang dilatih |
| RM-c | Frozen Encoder + RAC | Encoder beku diperkuat retrieval FAISS |

Model dasar `indobenchmark/indobert-base-p2`. Label biner (0 = normal,
1 = promosi judi). Metrik utama F1-macro.

## Lingkungan

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Python 3.13, torch 2.12.1, transformers 5.12.1 (lihat `requirements.txt` untuk
pin lengkap — versinya adalah yang benar-benar terpasang dan tervalidasi).
Notebook dijalankan dari folder `notebooks/` karena seluruh path relatif `../`.

## Menjalankan

Seluruh pipeline dipanggil dari notebook. Tidak ada `scripts/`, tidak ada CLI,
tidak ada UI, tidak ada test otomatis.

```
01_eda → 02_preprocessing → 03a_rma → 03b_rmb → 03c_rmc
       → 05_final_benchmark → 06_artifacts → 07_archive
```

Jangan lewati `02_preprocessing.ipynb`: notebook model bergantung padanya.

03a sampai 03c adalah kampanye sesungguhnya, satu notebook per skenario, dalam
satu mesin dan berurutan: 03a grid RM-a, 03b grid RM-b, 03c grid RM-c (seluruh
head RM-b x alpha x k). Ketiganya menulis ke `outputs/tuning/`, folder yang juga
dibaca 05, 06, dan 07. 03b menyimpan state setiap head di
`checkpoints/rmb_heads/` dan cache embedding di `features/`; 03c memuat keduanya,
tidak melatih ulang. Tidak ada resume otomatis: bila terputus, potong daftar grid
sesuai baris yang sudah tercatat di `runs_*.csv`.

Urutan penting di 03c: grid seluruh head → `decide_rmc_champion()` (default di
head RM-b resmi, penantang seluruh head, bootstrap berpasangan 10.000 iterasi) →
cek `weighting=uniform` → putusan ulang → `save_candidates()` menulis
`candidates.json` (kandidat #1 dan #2 tiap skenario, cap waktu dan hash) SEBELUM
test dibuka. 05 berhenti bila lingkungan berbeda dari saat tuning
(`verify_final_session`) atau hash kandidat tidak cocok (`load_candidates`).

`06_artifacts.ipynb` (setelah 05) membangkitkan Gambar 4.1-4.8 dan data mentah
Tabel 4.1-4.19 serta L.1 (satu sheet per tabel di `artifacts/data_tabel.xlsx`);
tabel siap tempel TIDAK dibuat, disusun manual saat penulisan.
`07_archive.ipynb` dijalankan terakhir supaya artefak ikut terbungkus arsip.
Nomor 04 sengaja kosong.

Checkpoint tidak ikut git dan tidak bisa dipulihkan tanpa mengulang tuning, jadi
03a sampai 05 dijalankan di mesin yang sama.

## Arsitektur

### Alur data

```
data/raw/data_labeling.csv
    → 02: decode entitas HTML, kunci NFKC, konflik label, dedup, split, clean_text, guard, gate checksum
    → data/interim/data_clean.csv, data/processed/{train,val,test}.csv + metadata.json
    → 03a: runs_rma.csv, rma_best.pt, rma_top/, hardware.json
    → 03b: features/, runs_rmb.csv, rmb_heads/, rmb_best.pt
    → 03c: runs_rmc.csv, rmc_best.pt, rmc_champion_decision.json, candidates.json
    → 05: metrics/, final_session di hardware.json, models/
    → 06: artifacts/gambar/, artifacts/data_tabel.xlsx
    → 07: outputs/faiss_index/, hasil_*.tar.gz
```

### Notebook

| Notebook | Isi |
|---|---|
| `01_eda` | EDA; mengunci kunci dedup, `max_length`, class weight, placeholder |
| `02_preprocessing` | Tahap 1-11: load, decode entitas HTML, NFKC, dedup, split, `clean_text`, guard anti-kebocoran, class weight, gate checksum, simpan, statistik [UNK] |
| `03a_rma_finetune` | Gate hardware, tokenisasi, `train_rma` (akumulasi gradien, AMP), pencatatan run, kalibrasi, grid tahap 1-2 |
| `03b_rmb_frozen` | Ekstraksi fitur beku (cache), `train_rmb`, pencatatan run, grid RM-b satu sel per tahap (1, 1B, 2, 3) |
| `03c_rmc_rac` | RAC, grid seluruh head x alpha x k, ringkasan per head, putusan juara, cek uniform, kandidat |
| `05_final_benchmark` | Gate lingkungan, evaluasi test, kandidat #2, benchmark latency per komponen, perbandingan, kriteria sukses |
| `06_artifacts` | Gambar 4.1-4.8 (PNG 300 dpi + PDF, koma desimal, pembulatan setengah ke atas) dan data Tabel 4.1-4.19, L.1 |
| `07_archive` | Biaya indeks FAISS lintas k, arsip `outputs/` |

### Mekanisme RAC (RM-c)

RAC menggabungkan dua **distribusi probabilitas** saat inferensi, bukan logit:

```
p_bert  = softmax(head(embedding))
p_retr  = distribusi label k tetangga terdekat (indeks FAISS HANYA dari train)
p_final = (1 - alpha) * p_bert + alpha * p_retr
pred    = argmax(p_final)
```

**Softmax diterapkan sekali, di cabang BERT sebelum fusi, tidak pernah
sesudahnya.** Kedua masukan sudah berupa distribusi dan bobot fusi berjumlah
satu, jadi `p_final` sudah sah; softmax kedua akan meratakan selisih dan bisa
mengubah argmax pada kasus nyaris seri. Saat menulis Bab 4, nyatakan fusinya di
level probabilitas — fusi level probabilitas dan level logit tidak ekuivalen.

Itu SATU-SATUNYA rumus RM-c. Rumus 1-4 (fusi level skor, alpha adaptif,
geometric pooling) sempat diuji lalu dikeluarkan dari kampanye (2026-09-24);
kodenya sudah dihapus saat refactor ke notebook (2026-09-30).

## Aturan yang tidak boleh dilanggar

1. **Angka efisiensi hanya sah dari satu hardware dan satu sesi.** Waktu latih,
   latency, dan peak memory tidak boleh dicampur lintas mesin atau sesi dalam
   satu tabel. F1 tidak bergantung hardware.
2. **Split test hanya dibuka di `05_final_benchmark.ipynb`.** Seleksi
   hyperparameter memakai validation. 03a-03c tidak pernah membaca embedding
   atau label test untuk evaluasi.
3. **Indeks FAISS hanya dari split train.** Memasukkan val atau test membuat
   retrieval menemukan sampel uji di dalam indeksnya sendiri.
4. **Jangan menimpa `data/processed/` tanpa gate checksum.** Tahap 9 notebook 02
   membandingkan checksum ISI split baru dengan yang lama (`check_split`) dan
   menolak melanjutkan bila berbeda. Yang dibandingkan isi kanonik, bukan SHA-256
   byte berkas: hash byte berbeda antar OS (CRLF vs LF) dan antar git `autocrlf`.
   Split yang berubah membuat seluruh riwayat run menjadi yatim.
5. **Input model adalah kolom `text_clean`**, bukan `textOriginal`.
6. **Model wajib `resize_token_embeddings`** karena preprocessing menambahkan
   `[URL]`, `[MENTION]`, `[NUM]`. Sudah ditangani `build_finetune_model` dan
   `build_encoder`, yang juga mengisi embedding baru dengan rata-rata subword
   kata benih (`SEED_WORDS`).

## Konvensi hasil

Seluruh keluaran kampanye di `outputs/tuning/`. `outputs/` di-gitignore (kecuali
figur EDA dan README): hasil run tidak disimpan di git. Amankan hasil dengan sel
"Arsipkan hasil" di `07_archive.ipynb` sebelum instance dihancurkan.

- `runs_{rma,rmb,rmc}.csv` — satu baris per run, menumpuk lintas sesi. Memuat
  kolom `catatan` (alasan konfigurasi dicoba) dan kolom turunan
  `delta_vs_best_f1_macro_pp`, `is_tie_with_best`, `overfit_signal`.
- `best.json` — juara per skenario menurut F1-macro validation; juara RM-c
  ditetapkan ulang oleh putusan di 03c (`decided=True`).
- `history/{scenario}_history.csv` — kurva per-epoch RM-a dan RM-b, kolom `run_id`.
- `checkpoints/` — `rma_best.pt`, `rma_top/` (tiga run RM-a teratas, bergulir,
  untuk kandidat #2), `rmb_heads/run_{id}.pt` (SETIAP head RM-b), `rmb_best.pt`,
  `rmc_best.pt` (head yang dipakai juara RM-c beserta konfigurasi fusinya).
- `features/<encoder>/` — embedding beku ketiga split dan `extract_meta.json`.
- `hardware.json` — lingkungan (GPU, VRAM, driver, CUDA, CPU, RAM, OS, versi
  Python/torch/transformers/faiss, seed), `tuning_sessions` (waktu catat dan
  boot), dan `final_session`.
- `rmc_champion_decision.json` — default, penantang, bootstrap, putusan, catatan biaya.
- `candidates.json` — kandidat #1/#2 tiap skenario, cap waktu, `content_sha256`.
- `metrics/` dari 05: `final_comparison.csv`, `final_predictions.csv`,
  `candidates_test.csv`, `inference_benchmark.csv`, `latency_breakdown.csv`,
  `index_stats.json`, `success_criteria.csv`.
- `artifacts/gambar/` dan `artifacts/data_tabel.xlsx` dari 06.
- `outputs/preprocessing/preprocessing_stats.json` dari 02 (di luar `metadata.json`
  supaya tidak menyentuh area gate checksum).

Hasil resmi Bab 4 adalah kampanye RTX 4090 (2026-09-30) di `outputs/tuning/` lokal;
notebook hasil eksekusinya di `outputs/notebooks_eksekusi/`. Ringkasan angka dan
catatan prosesnya di `PROGRESS.md`.

## Rancangan eksplorasi hyperparameter

**Metode hibrida** — grid kombinatorial untuk sumbu yang saling terkait,
coordinate descent untuk yang independen. Pembagian ini berasal dari mekanika
sesungguhnya di `train_rma` (03a): `steps = ceil(N/batch) x epochs`, warmup
adalah RASIO dari total langkah sehingga menormalkan dirinya sendiri, dan weight
decay AdamW memang terpisah dari gradien.

- **Grup A, digrid bersama:** `lr x epochs x batch`. Ketiganya bersama-sama
  menentukan lintasan optimasi — batch 32 pada 5 epoch memberi separuh jumlah
  langkah pembaruan dibanding batch 16, sehingga mengubah satu menggeser optimum
  yang lain.
- **Grup B, coordinate descent di sel pemenang:** `warmup_ratio` dan
  `weight_decay`.

**Aturan seleksi:** `val_f1_macro` utama, `val_f1_judi` sebagai pemecah seri
(kelas minoritas yang bisa tertutupi rata-rata makro). Selisih di bawah 0,15 pp
dihitung seri (satu seed) sehingga konfigurasi yang lebih murah menang. Baca grid
sebagai permukaan: pivot `lr x epochs` per `batch` memperlihatkan apakah lr
optimal ikut bergeser. Pemenang di tepi grid adalah sinyal untuk melebarkan
rentang, bukan untuk mengunci.

Rancangan lengkap ada di `tuning_grids/` (`.md` untuk penalaran, `.csv` siap
dimuat notebook 03a-03c).

**RM-c adalah satu grid: seluruh head RM-b x `alpha x k`.** Head dipilih lewat
`rmb_run_id` di konfigurasi RM-c, dan 66 konfigurasi `alpha x k` di
`RMC_TUNING_GRID.csv` diterapkan ke setiap head; setiap kombinasi satu baris
`runs_rmc.csv`. Juara RM-c TIDAK dipilih mekanis: default adalah konfigurasi
terbaik pada head RM-b resmi, dan penantang seluruh head hanya menggantikannya
bila selisihnya > 0,15 pp DAN batas bawah CI95 bootstrap berpasangan > 0
(`decide_rmc_champion`). Bila penantang menang dengan head lain, biaya RM-c
adalah biaya head itu (`head_is_official_rmb=False`). Tahap 2 mencoba
`weighting=uniform` sekali di sel juara, lalu putusan dibuat ulang. RM-c tidak
melatih apa pun, tetapi biayanya adalah biaya head yang dipakainya (bukan nol)
ditambah retrieval saat inferensi.

## Dataset

`data/raw/data_labeling.csv` — komentar YouTube berlabel. Kolom yang dipakai
`textOriginal` dan `label`; sisanya metadata dan sebagiannya memuat identitas.

`data/processed/` berisi split 70:15:15 (train 6.587 / val 1.402 / test 1.405,
total 9.394, sekitar 18% kelas judi, rasio 4,5:1). Class weight ada di
`metadata.json` (`{0: 0.611, 1: 2.754}`) dan dipakai pada `CrossEntropyLoss`.
Karena timpang, laporkan F1-macro dan F1 kelas judi, jangan accuracy saja.

## Kriteria sukses

Strategi ringan (RM-b atau RM-c) dianggap kompetitif bila memenuhi minimal dua
dari tiga syarat:

- Selisih F1 tidak lebih dari 3 poin persentase terhadap RM-a
- Pengurangan trainable parameter minimal 90%
- Pengurangan waktu latih minimal 50%

## Struktur branch

Branch: `main` (codebase rujukan lama berbasis `src/`, tanpa data hasil run),
`local` (kampanye di mesin lokal, RTX 3050 Laptop), `vast.ai` (kampanye lama di
instance Vast.ai, RTX 3090), dan `refactor/notebook-style` (codebase notebook-only
dan split terbaru, titik awal kampanye ulang di RTX 4090; langkah instance ada di
README). Aturan #1 di atas berlaku per branch dan per mesin: jangan campur angka
efisiensi lintas branch dalam satu tabel Bab 4. Hasil run tidak disimpan di git.

## Status

Lihat `PROGRESS.md` sebagai sumber kebenaran.
