# AUDIT PIPELINE — Verifikasi Klaim Naskah vs Kode

Audit **read-only** terhadap kode yang benar-benar dijalankan untuk skripsi
"Analisis Trade-off Performa dan Efisiensi Komputasi pada Strategi Adaptasi IndoBERT dengan
Retrieval-Augmented Classification untuk Deteksi Komentar Promosi Judi Daring".

Tanggal audit: 2026-08-16 · Basis: `HEAD` (commit `4a1887c`) + isi `results/vast/` (hasil kampanye
2026-07-25, RTX 3090). Tidak ada kode yang diubah. Tidak ada training/GPU yang dijalankan; eksekusi
terbatas pada fungsi murni (`clean_text`, `rac.build_faiss_index`, `rac.retrieve`) di CPU.

---

## 0. Peta repositori (skrip yang relevan)

| Peran | Berkas | Catatan |
|---|---|---|
| Pra-pemrosesan (fungsi) | `src/preprocessing.py` | `normalize_nfkc`, `clean_text`, `deduplicate`, `stratified_split`, `compute_class_weights` |
| Pra-pemrosesan (eksekutor) | `notebooks/02_preprocessing.ipynb` | satu-satunya skrip yang **menulis** `dataset/splits/` + `metadata.json` |
| Tokenisasi & Dataset | `src/dataset.py` | `load_tokenizer` (daftar special token), `GamblingCommentDataset` |
| Pabrik model & ekstraksi fitur | `src/modeling.py` | `build_finetune_model` (RM-a), `build_encoder` (beku), `mean_pool`, `extract_features`, `FrozenHead`/`MLPHead` |
| Engine latih/eval 1-konfigurasi | `src/tuning.py` | `train_eval_rma`, `train_eval_rmb`, `eval_rmc`, `RunLogger`, `test_rma` |
| Worker/orkestrator (JALUR UTAMA Bab 4) | `src/job_runner.py` | fase `rma`/`rmb`/`rmc`/`final`, `ensure_features` (ekstraksi fitur), `phase_final` (benchmark satu sesi) |
| RAC / indeks FAISS | `src/rac.py` | `l2_normalize`, `build_faiss_index`, `retrieve`, `retrieval_distribution`, `softmax`, `fuse`, `rac_predict` |
| Metrik & efisiensi | `src/evaluate.py` | `classification_metrics`, `count_parameters`, `peak_gpu_mem_mb`, `measure_latency` |
| UI kendali penalaan | `app.py` | Streamlit 4 tab; hanya menulis `job.json` lalu memanggil `src/job_runner.py` |
| Reproduksi lokal (TIDAK dipakai Bab 4) | `run_local_training.py` | menulis ke `results/local/` — folder ini **kosong** |
| Benchmark modul FAISS terpisah | `scripts/benchmark_faiss_index.py` | menulis `results/faiss_index/` (diukur di laptop/CPU, bukan sesi 3090) |
| Notebook lama | `notebooks/03a`, `03b`, `03c`, `04a*`, `04b*` | legacy; memanggil fungsi `src/` yang sama, bukan sumber angka Bab 4 |

**Tidak ada skrip pembangunan indeks FAISS yang berdiri sendiri untuk produksi.** Indeks dibangun
*on-the-fly* di dalam memori setiap kali RM-c dievaluasi (`src/tuning.py:242`,
`src/job_runner.py:497`). Berkas `results/faiss_index/train_index.faiss` hanya artefak benchmark.

---

## 1. Tabel ringkasan

| Kode | Vonis | Ringkasan satu baris |
|---|---|---|
| **A1** | **SEBAGIAN** | Sama-sama `perf_counter` dan sama-sama mengecualikan pemuatan model, tetapi cakupannya tidak identik: 81,10 s memuat 5× evaluasi validasi ber-encoder penuh + salinan 110 M bobot; 11,65 s memuat ekstraksi 3 split (termasuk **test**) yang diukur **sekali** lalu dipakai ulang di semua run. |
| **A2** | **SESUAI** | `resize_token_embeddings` dipanggil di satu tempat (`_resize_and_init`) yang dilewati RM-a **dan** encoder beku; ekstraksi fitur, indeks FAISS, dan benchmark inferensi semuanya memakai jalur itu. |
| **A3** | **SESUAI** | `IndexFlatIP` + L2-normalisasi manual di **kedua** sisi (add & search), tidak in-place; similaritas teramati 0,4161–1,0000. |
| **B1** | **SESUAI** | `clean_text` = buang karakter tak-terlihat → NFKC → `[URL]` → `[MENTION]` → `[NUM]` → rapikan whitespace. **TIDAK ADA** normalisasi slang/kata tidak baku/singkatan. |
| **B2** | **SESUAI** | `\b\d+\b` — `DORA77`, `slot88`, `x500`, `situs777.com` semua utuh (bukti eksekusi di §B2). |
| **B3** | **SEBAGIAN** | Rantai sebenarnya menyisipkan **split** dan **`clean_text`** di antara tahap 3 dan 4; guard bukan penyaring global melainkan operasi pasca-split pada `text_clean`, hanya di val/test. |
| **B4** | **TIDAK SESUAI** (klaim Bab 3) | Tidak ada filter panjang minimum, tidak ada deteksi bahasa, tidak ada pembuangan non-teks. Klaim Bab 4 (3 tahap) yang benar. |
| **B5** | **SESUAI** | `train_test_split` 2 tahap, `stratify=label`, seed 42 hardcoded di `preprocessing.RANDOM_SEED`; ketiga skenario membaca `dataset/splits/*.csv` yang sama. |
| **B6** | **SESUAI** | `compute_class_weight('balanced')` dari train saja; tensor bobot yang **sama persis** dipakai RM-a dan RM-b; tidak ada oversampling/SMOTE. Cabang retrieval RM-c memang tanpa pembobotan kelas. |
| **C1** | **TIDAK SESUAI** | Angka adalah **MiB** (`max_memory_allocated()/1024**2`) dilabeli MB → 3.091,8 MiB = 3.242,0 MB dan 63,9 MiB = 67,0 MB; selain itu 63,9 **tidak** mencakup fase ekstraksi fitur (529,5 MiB, tercatat terpisah). |
| **C2** | **SEBAGIAN** | Harness identik (10 warmup, 100 run, rata-rata, batch 1, tokenisasi di luar timer, `eval()`+`no_grad`, `synchronize` mengapit blok), tetapi RM-c menyisipkan sinkronisasi device per-iterasi (`.cpu().numpy()`) dan mengambil vektor kueri dari cache, bukan dari keluaran encoder yang baru dihitung. |
| **C3** | **TIDAK SESUAI** | Representasi bukan token `[CLS]` melainkan **masked mean-pooling** (`src/modeling.py:78-87`). Sisanya benar: 6.588 vektor train-only, 768-d, float32, artefak yang sama dengan masukan head. |
| **C4** | **SESUAI** | Softmax hanya di cabang BERT sebelum fusi; `fuse` → `argmax` langsung, tanpa softmax kedua; ada fallback uniform. |
| **C5** | **SEBAGIAN** | Artefak final (`checkpoints/rmc_best.pt`, `best.json`) berisi `alpha=0.2, k=5, weighting=similarity` — sesuai naskah — tetapi field `run_id`-nya (13) menunjuk baris `runs_rmc.csv` yang justru `k=1`. Konfigurasi benar, penomoran run tidak konsisten. |
| **C6** | **SESUAI** (TIDAK ada pemakaian test) | `eval_test` default OFF, 124 baris run penalaan tidak punya satu pun kolom `test_*`, grid CSV tidak punya kolom `eval_test`, seleksi & best-epoch murni `val_f1_macro`. |

---

## 2. Detail per pemeriksaan

### A1 — Cakupan pengukuran `train_time_s` pada RM-a

**Klaim naskah:** RM-a 81,10 s vs RM-b 11,65 s → penghematan 85,64%; 11,65 s = 7,90 s ekstraksi
fitur + 3,75 s pelatihan head.

**Angka di artefak (terverifikasi):** `results/vast/runs_rma.csv` run #13 → `train_time_s=81.1`;
`results/vast/runs_rmb.csv` run #28 → `extract_time_s=7.9`, `head_train_time_s=3.75`,
`train_time_s=11.65`; `results/vast/metrics/success_criteria.csv` → `time_reduction_pct=85.64`.

**Timer RM-a** — `src/tuning.py:139` (`t0 = time.perf_counter()`) sampai `src/tuning.py:165`
(`"train_time_s": round(time.perf_counter() - t0, 1)`). Yang berada **di dalam** rentang itu:

```python
# src/tuning.py:139-165 (ringkas)
t0 = time.perf_counter()
for ep in range(1, cfg["epochs"] + 1):
    model.train(); ...                         # loop training (tokenisasi on-the-fly ikut di sini)
    yv, pv, _ = _predict(model, va, device)    # 155: evaluasi validasi TIAP epoch (encoder penuh)
    ...
    if vm["f1_macro"] > best_f1:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}   # 164: salin 109,5 juta parameter ke CPU
extra = {..., "train_time_s": round(time.perf_counter() - t0, 1), ...}   # 165
```

- **Bukan hanya loop training.** Evaluasi validasi per epoch (`src/tuning.py:155`, 1.402 sampel,
  forward encoder penuh, 5 epoch) berada di dalam timer.
- **Salinan checkpoint** `state_dict` 109.485.314 parameter ke CPU (`src/tuning.py:164`) juga di
  dalam timer, terjadi setiap kali val F1 membaik (run #13: `best_epoch=4`).
- **Di luar timer:** pemuatan model (`src/tuning.py:128`), pembuatan `DataLoader`
  (`src/tuning.py:131`), pembuatan optimizer/scheduler/`GradScaler` (`src/tuning.py:133-136`),
  serta `reset_peak_memory_stats` (`src/tuning.py:138`).
- **Tokenisasi ikut terhitung** untuk RM-a: `GamblingCommentDataset.__getitem__`
  (`src/dataset.py:88-94`) mentokenisasi *on-the-fly* di dalam iterasi DataLoader, jadi setiap epoch
  mentokenisasi ulang seluruh train + val.

**Timer RM-b** terdiri atas dua komponen terpisah yang dijumlahkan di `src/job_runner.py:275`
(`"train_time_s": round(extra["train_time_s"] + (ex_meta.get("extract_time_s") or 0), 2)`):

1. `extract_time_s` — `src/job_runner.py:139` → `152`. Mencakup: pembuatan `Dataset` + `DataLoader`,
   tokenisasi, forward encoder beku, mean-pooling, dan `np.save` untuk **ketiga** split
   (`train`, `val`, **`test`**) — `src/job_runner.py:143-149`. **Di luar**: pembangunan encoder
   (`src/job_runner.py:138`). Nilainya disimpan ke `extract_meta.json` (`src/job_runner.py:150-154`)
   dan **dibaca ulang apa adanya** oleh setiap run RM-b (`src/job_runner.py:267,274`) — karena itu
   ke-31 baris `runs_rmb.csv` memuat `extract_time_s = 7.9` yang identik; angka itu **diukur satu
   kali**, bukan per konfigurasi.
2. `head_train_time_s` — `src/tuning.py:192` → `211`. Mencakup loop epoch head, evaluasi validasi
   per epoch (`src/tuning.py:201-203`, di atas fitur cache), dan salinan `best_state` head
   (`src/tuning.py:210`). **Di luar**: pemindahan tensor fitur ke GPU (`src/tuning.py:185-186`),
   pemuatan `.npy` dari disk (`src/job_runner.py:158`), dan pengukuran latency (`src/tuning.py:214`,
   setelah timer berhenti).

**Mekanisme waktu:** keduanya `time.perf_counter()` (`src/tuning.py:139`, `:165`, `:192`, `:211`;
`src/job_runner.py:139`, `:152`). Tidak ada perbedaan sumber jam.

**`torch.cuda.synchronize()` sebelum timer berhenti: TIDAK ADA** — tidak dipanggil di
`train_eval_rma`, `train_eval_rmb`, maupun `ensure_features`. Satu-satunya `synchronize()` di
repositori ada di `src/evaluate.py:140,145` (khusus `measure_latency`). Sinkronisasi hanya terjadi
secara *implisit*: `.cpu().numpy()` di `_predict` (`src/tuning.py:116-117`), `.detach().cpu()` pada
salinan checkpoint (`src/tuning.py:164`, `:210`), dan `.cpu().numpy()` di `extract_features`
(`src/modeling.py:110`). Kebetulan ketiga jalur berakhir dengan operasi seperti itu, sehingga
kesalahannya kecil — tetapi itu efek samping, bukan jaminan eksplisit.

**Vonis: SEBAGIAN.** Kedua angka memakai jam yang sama dan sama-sama mengecualikan pemuatan bobot
pra-latih, sehingga keduanya bisa dibaca sebagai "biaya memperoleh model terlatih dari CSV split".
Namun cakupannya tidak simetris di tiga titik:

| Komponen | RM-a (81,10 s) | RM-b (11,65 s) |
|---|---|---|
| Tokenisasi | ikut, diulang tiap epoch (5×) | ikut, sekali (di dalam 7,90 s) |
| Forward encoder atas **val** | 5× (evaluasi tiap epoch) | 1× (saat ekstraksi) |
| Forward encoder atas **test** | tidak pernah | **ikut** (ekstraksi 3 split) |
| Salinan checkpoint | 109,5 juta parameter → CPU | 789 ribu parameter → CPU |
| Sifat angka | diukur ulang tiap run | 7,90 s adalah konstanta hasil satu pengukuran, dipakai ulang di 31 run |

Ketimpangan terbesar ada di sisi RM-a (5 kali evaluasi validasi ber-encoder penuh + penyalinan bobot
besar); ketimpangan di sisi RM-b (ikut mengekstraksi test set) berlawanan arah dan jauh lebih kecil.
Naskah tidak bisa menyatakan keduanya "mengukur hal yang sama persis" tanpa menyebutkan bahwa
angka RM-a mencakup validasi per-epoch dan angka RM-b mencakup ekstraksi ketiga split.

---

### A2 — Pemanggilan `resize_token_embeddings()`

**Klaim naskah:** tiga special token didaftarkan, dan setiap model wajib memanggil
`resize_token_embeddings()` sebelum pelatihan maupun inferensi.

**Pendaftaran token** — hanya satu tempat:

```python
# src/dataset.py:59
tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
```

`SPECIAL_TOKENS = ["[URL]", "[MENTION]", "[NUM]"]` (`src/preprocessing.py:37`, diimpor di
`src/dataset.py:21`). Setiap pemuatan tokenizer di pipeline melewati `load_tokenizer`:
`src/job_runner.py:118` dan `run_local_training.py:51`. Tidak ada `AutoTokenizer.from_pretrained`
langsung di jalur pelatihan/inferensi selain di dalam `load_tokenizer` itu sendiri
(`src/dataset.py:50,54`).

**Pemanggilan resize** — juga hanya satu tempat, lalu dipakai bersama:

```python
# src/modeling.py:52-55
def _resize_and_init(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    init_special_token_embeddings(model, tokenizer)
    return model
```

- RM-a: `build_finetune_model` → `_resize_and_init(model, tokenizer)` — `src/modeling.py:63`.
- Encoder beku (RM-b/RM-c): `build_encoder` → `_resize_and_init(encoder, tokenizer)` —
  `src/modeling.py:73`, baru kemudian `requires_grad_(False)` dan `eval()` (`:74-75`).

**Titik rawan yang diminta (ekstraksi fitur RM-b): AMAN.** `ensure_features` membangun encoder
lewat `M.build_encoder(...)` (`src/job_runner.py:138`) — jadi resize + inisialisasi terjadi
**sebelum** `requires_grad_(False)`, bukan dilewati. Tambahan: baris embedding baru tidak diisi
noise acak melainkan rata-rata sub-word kata seed (`tautan`/`akun`/`angka`) — `src/modeling.py:26-49`
— yang relevan justru karena encoder beku tidak akan pernah melatih baris tersebut.

**Indeks FAISS RM-c:** indeks dibangun dari array `.npy` hasil ekstraksi di atas
(`src/tuning.py:242`, `src/job_runner.py:497`) — **tidak menyentuh tokenizer sama sekali**, sehingga
otomatis mewarisi tokenizer yang sudah diperluas. Konsistensi encoder juga dijaga eksplisit: RM-c
menolak berjalan bila `model_name` job berbeda dari encoder head RM-b (`src/job_runner.py:339-345`).

**Evaluasi & pengukuran latency:** `phase_final` memakai `ctx["tokenizer"]` yang sama
(`src/job_runner.py:517-518`) dan membangun ulang model/encoder lewat `build_finetune_model` /
`build_encoder` (`src/job_runner.py:492`, `:495`) — keduanya lewat `_resize_and_init`.

| Komponen | Vonis |
|---|---|
| RM-a (pelatihan) | **SESUAI** (`src/modeling.py:63`) |
| RM-b (pelatihan head) | **SESUAI** — head bekerja di atas fitur; encoder yang menghasilkan fitur sudah di-resize |
| Ekstraksi fitur | **SESUAI** (`src/job_runner.py:138` → `src/modeling.py:73`) |
| Pembangunan indeks FAISS | **SESUAI** (memakai artefak fitur; tanpa tokenizer sendiri) |
| Inferensi / benchmark latency | **SESUAI** (`src/job_runner.py:492,495,517`) |

**Vonis: SESUAI.**

---

### A3 — Normalisasi L2 pada indeks dan kueri FAISS

**Klaim naskah:** flat index dengan metrik inner product atas vektor ternormalisasi ⇒ setara cosine.

```python
# src/rac.py:31-56
def l2_normalize(x, eps=1e-12):
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)          # menghasilkan ARRAY BARU, bukan in-place

def build_faiss_index(train_emb):
    emb = l2_normalize(train_emb)                 # 46: normalisasi SISI INDEKS
    index = faiss.IndexFlatIP(emb.shape[1])       # 47
    index.add(emb)                                # 48

def retrieve(index, query_emb, k):
    q = l2_normalize(query_emb)                   # 54: normalisasi SISI KUERI
    sims, idx = index.search(q, k)                # 55
```

- **Jenis indeks:** `faiss.IndexFlatIP` (`src/rac.py:47`) — eksak, inner product.
- **Sebelum `add`:** ya (`src/rac.py:46`). **Sebelum `search`:** ya (`src/rac.py:54`).
- `faiss.normalize_L2()` **tidak dipakai**; normalisasi manual dengan numpy. Ini justru menghindari
  jebakan `faiss.normalize_L2` yang bekerja in-place.
- **Risiko normalisasi ganda: tidak ada.** `x / np.clip(...)` selalu membuat array baru, jadi array
  masukan tidak termutasi. Diverifikasi dengan menjalankan fungsi asli terhadap
  `results/vast/features/.../train_emb.npy`: setelah `build_faiss_index`, norma baris array asli
  tetap 19,26 / 16,87 / 20,55 (belum ternormalisasi), sedangkan array yang masuk indeks bernorma
  1,0000. Lagipula normalisasi L2 idempoten, sehingga normalisasi berulang pun tidak mengubah nilai.
- **Rentang nilai kemiripan (dieksekusi, embedding test → indeks train, k=5):**
  min 0,4161, maks 1,0000, rata-rata 0,7793; **0%** nilai ≤ 0. Konsisten dengan rentang cosine
  [-1, 1] dan dengan `mean_top1_sim` yang tercatat di `results/faiss_index/faiss_search_latency.csv`.

**Vonis: SESUAI.** Klaim kesetaraan dengan cosine similarity sah — normalisasi diterapkan di **kedua**
sisi. Satu koreksi redaksional untuk naskah: normalisasinya manual (`src/rac.py:31-35`), bukan
`faiss.normalize_L2()`.

---

### B1 — Isi fungsi `clean_text`

**Isi lengkap, apa adanya** (`src/preprocessing.py:63-80`):

```python
_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|t\.me/\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_NUM_RE = re.compile(r"\b\d+\b")          # hanya angka berdiri sendiri
_WS_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """NFKC -> placeholder URL/mention/angka -> rapikan whitespace. ..."""
    t = normalize_nfkc(text)
    t = _URL_RE.sub(URL_PLACEHOLDER, t)
    t = _MENTION_RE.sub(MENTION_PLACEHOLDER, t)
    t = _NUM_RE.sub(NUM_PLACEHOLDER, t)
    return _WS_RE.sub(" ", t).strip()
```

dengan (`src/preprocessing.py:43-58`):

```python
_INVIS_CP = [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0x00AD, *range(0xFE00, 0xFE10)]
_INVISIBLE = re.compile("[" + "".join(chr(c) for c in _INVIS_CP) + "]")

def normalize_nfkc(text) -> str:
    return unicodedata.normalize("NFKC", _INVISIBLE.sub("", str(text)))
```

**Urutan transformasi yang benar-benar dijalankan:**

1. `str(text)` — konversi paksa ke string (`src/preprocessing.py:58`).
2. Buang karakter tak-terlihat: ZWSP, ZWNJ, ZWJ, LRM, RLM, word joiner, soft hyphen, variation
   selector VS1–VS16 (`src/preprocessing.py:43-48,58`).
3. Normalisasi Unicode **NFKC** (`src/preprocessing.py:58`).
4. URL → `[URL]`; pola: `https?://…`, `www.…`, **dan `t.me/…`**, case-insensitive
   (`src/preprocessing.py:63,77`).
5. Mention `@\w+` → `[MENTION]` (`src/preprocessing.py:64,78`).
6. Angka berdiri sendiri `\b\d+\b` → `[NUM]` (`src/preprocessing.py:65,79`).
7. Runtun whitespace → satu spasi, lalu `.strip()` (`src/preprocessing.py:66,80`).

**Pertanyaan kunci — normalisasi kata tidak baku / slang / pemetaan singkatan: TIDAK.**
Tidak ada kamus, tabel pemetaan, maupun aturan penggantian kata di seluruh jalur pra-pemrosesan.
Pencarian `slang|baku|kamus|normalisasi kata` pada `src/`, `app.py`, `run_local_training.py`, dan
`scripts/` tidak menghasilkan satu pun kecocokan. Satu-satunya struktur mirip kamus di repositori
adalah `_SEED_WORDS` (`src/modeling.py:26-30`) yang berisi 3 entri dan dipakai untuk
**menginisialisasi embedding special token**, bukan untuk mengubah teks.

**`str.lower()` / `casefold()` di jalur pra-pemrosesan: TIDAK ADA.** Satu-satunya `.lower()` di
repositori berada di `run_local_training.py:236` dan hanya menyusun nama berkas CSV. Perlu dicatat
untuk naskah: lowercasing tetap terjadi, tetapi **di level tokenizer** — `indobert-base-p2` adalah
model *uncased* (`do_lower_case=True`, didokumentasikan di `src/dataset.py:4`) — bukan di
`clean_text`.

**Transformasi yang tidak disebut dalam klaim naskah tetapi benar-benar ada:**

- pembuangan karakter tak-terlihat sebelum NFKC (langkah 2);
- kolaps whitespace + `strip()` (langkah 7);
- URL regex juga menangkap `www.…` dan `t.me/…`, tidak hanya `http(s)://`.

**Transformasi yang diklaim TIDAK dilakukan — terkonfirmasi TIDAK ADA:** lowercasing manual,
penghapusan emoji, penghapusan tanda baca, pembuangan komentar pendek/non-Indonesia/outlier panjang.
`dataset/processed/metadata.json` mencatatnya eksplisit: `"kept_as_is": ["short_comments",
"non_indonesian", "length_outliers", "emoji"]`, `"lowercase_manual": false`.

**Vonis: SESUAI**, dengan tiga transformasi tambahan di atas yang belum tercermin di naskah.

---

### B2 — Pola regex untuk `[NUM]`

**Pola:** `_NUM_RE = re.compile(r"\b\d+\b")` — `src/preprocessing.py:65`.

**Hasil eksekusi `clean_text` terhadap string uji** (dijalankan dengan `.venv`, fungsi asli, tanpa
modifikasi):

| Masukan | Keluaran |
|---|---|
| `DORA77` | `DORA77` |
| `slot88 gacor` | `slot88 gacor` |
| `wa 08123456789` | `wa [NUM]` |
| `menang 5 juta` | `menang [NUM] juta` |
| `situs777.com daftar sekarang` | `situs777.com daftar sekarang` |
| `Cek @adminjudi ya` | `Cek [MENTION] ya` |
| `link https://bit.ly/abc123` | `link [URL]` |
| `PG SOFT x500` | `PG SOFT x500` |

**Vonis: SESUAI.** Nama brand alfanumerik benar-benar utuh. `DORA77` **tidak** berubah menjadi
`DORA[NUM]` — batas kata `\b` tidak ada di antara `A` dan `7` (keduanya karakter word), sehingga
digit yang menempel pada huruf tidak pernah tersentuh. Berlaku sama untuk `slot88`, `x500`, dan
`situs777` (digit diapit huruf di kiri dan titik di kanan).

Dua catatan tambahan untuk ketelitian naskah:

- `situs777.com` **juga tidak** menjadi `[URL]` — `_URL_RE` menuntut skema `http(s)://`, prefiks
  `www.`, atau `t.me/`; domain telanjang tidak tertangkap.
- `wa 08123456789` menunjukkan nomor kontak yang berdiri sendiri **hilang** menjadi `[NUM]`; ini
  konsekuensi desain yang benar menurut spesifikasi, tetapi berarti nomor WhatsApp tidak lagi
  membedakan satu komentar dari komentar lain.

---

### B3 — Urutan deduplikasi, pembagian data, dan guard anti-leakage

**Rantai operasi yang sebenarnya** (`notebooks/02_preprocessing.ipynb`, satu-satunya penulis
`dataset/splits/`):

```
baca CSV (cell 3)                                   14.237
  → dropna(['textOriginal','label']) (cell 3)       14.227
  → nfkc_key = NFKC(textOriginal) (src/preprocessing.py:105)
  → resolve konflik label (>1 label per key → 1) (src/preprocessing.py:85-95)
  → drop_duplicates(subset='nfkc_key', keep='first') (src/preprocessing.py:107)   9.412
  → stratified_split 70/15/15, seed 42 (cell 7 → src/preprocessing.py:113-134)    6.588 / 1.412 / 1.412
  → text_clean = clean_text(textOriginal) PER SPLIT (cell 9)
  → guard anti-leakage atas text_clean, hanya membuang dari val/test (cell 11)    6.588 / 1.402 / 1.405
  → simpan splits + data_clean.csv + metadata.json (cell 13, 15)                  total 9.395
```

**Posisi split:** persis **di antara** tahap 3 (dedup → 9.412) dan tahap 4 (guard → 9.395). Jadi
tahap 3 adalah operasi tingkat-korpus, sedangkan tahap 4 adalah operasi tingkat-split yang mustahil
dijalankan sebelum split ada. `clean_text` juga berada di celah itu (cell 9), sesuai klaim naskah
bahwa `clean_text` diterapkan setelah split.

**Kunci dedup:** `nfkc_key = normalize_nfkc(textOriginal)` (`src/preprocessing.py:105`) — **NFKC
saja**, tanpa lowercase, tanpa strip, tanpa kolaps whitespace, tanpa pembuangan tanda baca. Yang
ditambahkan hanyalah pembuangan karakter tak-terlihat yang menjadi bagian dari `normalize_nfkc`
(`src/preprocessing.py:58`). Kebijakan konflik label: grup dengan >1 label unik dipaksa ke label 1
**sebelum** dedup (`src/preprocessing.py:85-95`, 3 grup / 32 baris menurut `metadata.json`).

**Kunci guard:** kolom **`text_clean`** — yaitu teks **hasil** `clean_text`, bukan teks mentah:

```python
# notebooks/02_preprocessing.ipynb, cell 11
train_clean = set(train_df['text_clean'])
val_df  = val_df[~val_df['text_clean'].isin(train_clean)].reset_index(drop=True)
test_df = test_df[~test_df['text_clean'].isin(train_clean | set(val_df['text_clean']))].reset_index(drop=True)
```

Guard ini punya dua lapis: (a) `assert` bahwa `nfkc_key` tidak beririsan antar split (cell 11, selalu
0 karena dedup + partisi), dan (b) pembuangan tumpang tindih `text_clean` yang **baru muncul setelah
placeholder mengolapskan template-spam yang tadinya berbeda URL/angka**.

**Hanya val/test yang dibuang; train utuh.** Terkonfirmasi di kode di atas (baris train tidak pernah
difilter) dan di angka: `metadata.json` mencatat train `n=6588` — persis `round(9412×0,70)` — dengan
val 1.412→1.402 (−10) dan test 1.412→1.405 (−7), total 17.

**NFKC diterapkan dua kali** — sekali untuk `nfkc_key` (`src/preprocessing.py:105`) dan sekali lagi
di dalam `clean_text` (`src/preprocessing.py:76`). **Tidak menimbulkan perbedaan hasil**: NFKC
idempoten, dan `clean_text` tidak menerima keluaran dedup melainkan kembali membaca kolom mentah
`textOriginal` (cell 9), sehingga tidak ada komposisi normalisasi yang bertumpuk.

**Vonis: SEBAGIAN — penyajian empat tahap berurutan menyesatkan bila tanpa keterangan.** Empat angka
(14.237 → 14.227 → 9.412 → 9.395) semuanya benar dan tereproduksi dari `metadata.json`, tetapi
menampilkannya sebagai rantai penyaringan lurus menyembunyikan dua hal: (i) split dan `clean_text`
terjadi di antara langkah ketiga dan keempat, (ii) langkah keempat bukan penyaringan korpus melainkan
guard yang beroperasi pada representasi berbeda (`text_clean`, bukan `nfkc_key`) dan hanya mengenai
val/test. Konsekuensi yang perlu dinyatakan di naskah: 9.395 **bukan** ukuran korpus unik hasil
dedup, melainkan 9.412 dikurangi baris val/test yang menjadi kembar setelah placeholder diterapkan.

---

### B4 — Penyaringan yang benar-benar dijalankan

| Penyaringan | Ada? | Bukti |
|---|---|---|
| Panjang teks minimum | **TIDAK** | Tidak ada filter panjang di mana pun pada `src/preprocessing.py` atau `notebooks/02_preprocessing.ipynb`; satu-satunya operasi pembuangan baris adalah `dropna` (cell 3), `drop_duplicates` (`src/preprocessing.py:107`), dan guard (cell 11). `metadata.json` → `"kept_as_is": [..., "short_comments", ...]` |
| Deteksi bahasa / pembuangan non-Indonesia | **TIDAK** | Tidak ada pustaka deteksi bahasa di `requirements.txt` maupun impor apa pun di `src/`; `metadata.json` → `"kept_as_is": [..., "non_indonesian", ...]` |
| Pembuangan komentar non-teks | **TIDAK** | Tidak ada filter emoji/simbol/karakter; `clean_text` hanya mengganti, tidak pernah membuang baris (`src/preprocessing.py:69-80`); `metadata.json` → `"kept_as_is": [..., "emoji"]` |
| Pembuangan outlier panjang | **TIDAK** | `metadata.json` → `"kept_as_is": [..., "length_outliers", ...]`; truncation ke 128 token terjadi di tokenizer (`src/dataset.py:90-92`), yang memotong teks, bukan membuang baris |

**Vonis: klaim Bab 3 TIDAK SESUAI, klaim Bab 4 SESUAI.** Yang benar-benar mereduksi jumlah baris
hanya tiga: baris kosong/NaN, deduplikasi NFKC-exact, dan guard anti-leakage — persis seperti
rumusan Bab 4. Kalimat Bab 3 tentang "panjang teks tidak memadai" dan "komentar non-teks
dieliminasi" tidak punya padanan di kode.

---

### B5 — Stratified split dan seed

```python
# src/preprocessing.py:113-134
def stratified_split(df, label="label", seed=RANDOM_SEED, ratios=(0.70, 0.15, 0.15)):
    train_r, val_r, test_r = ratios
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios harus berjumlah 1.0"
    train_df, rest_df = train_test_split(
        df, train_size=train_r, stratify=df[label], random_state=seed)
    test_within_rest = test_r / (val_r + test_r)          # = 0,5
    val_df, test_df = train_test_split(
        rest_df, test_size=test_within_rest, stratify=rest_df[label], random_state=seed)
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True))
```

- **Fungsi:** `sklearn.model_selection.train_test_split` (`src/preprocessing.py:27`).
- **Stratified terhadap kolom label:** ya, di **kedua** panggilan (`stratify=df[label]`
  `src/preprocessing.py:124`; `stratify=rest_df[label]` `src/preprocessing.py:130`).
- **Cara mencapai 70:15:15:** dua panggilan berturut-turut — 70% vs sisa 30%, lalu sisa dibelah
  `test_size = 0,15/(0,15+0,15) = 0,5` (`src/preprocessing.py:127`). Hasilnya 6.588 / 1.412 / 1.412
  sebelum guard; setelah guard 6.588 / 1.402 / 1.405 (`metadata.json`).
- **Seed:** `RANDOM_SEED = 42`, **hardcoded** di `src/preprocessing.py:31`, dipakai sebagai nilai
  default parameter (`src/preprocessing.py:114`) dan diteruskan eksplisit oleh notebook
  (`seed=RANDOM_SEED`, cell 7). Tidak dibaca dari berkas konfigurasi/argumen CLI. Seed pelatihan
  (juga 42) terpisah dan berada di `src/tuning.py:37-41` serta `set_seed` (`src/tuning.py:49-51`).
- **Ketiga skenario membaca berkas split yang sama:** `build_ctx` memuat
  `dataset/splits/{train,val,test}.csv` satu kali per job (`src/job_runner.py:109-111`) dan
  membagikannya ke seluruh fase. RM-b/RM-c bahkan tidak menyentuh CSV lagi — keduanya memakai
  embedding `.npy` yang diekstraksi dari `ctx` yang sama (`src/job_runner.py:143-147`). **Tidak ada
  pemanggilan `stratified_split` atau `train_test_split` di luar `src/preprocessing.py`**; tidak ada
  skenario yang melakukan split ulang.

**Vonis: SESUAI.**

---

### B6 — Penerapan class weight

**Perhitungan** — `compute_class_weight('balanced')` dari sklearn, dihitung **hanya dari label
train**:

```python
# src/preprocessing.py:137-145
def compute_class_weights(y, classes=(0, 1)) -> dict:
    classes = np.array(classes)
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(y))
    return {int(c): float(w) for c, w in zip(classes, weights)}
```

Dipanggil `pp.compute_class_weights(train_df['label'])` (`notebooks/02_preprocessing.ipynb` cell 15)
dan disimpan ke `dataset/processed/metadata.json`:
`{"0": 0.6110183639398998, "1": 2.7518796992481205}` — cocok dengan angka naskah (0,6110 / 2,7519).
Nilai ini juga konsisten dengan rumus `balanced` atas distribusi train (5.391/1.197):
6588/(2×5391) = 0,61102 dan 6588/(2×1197) = 2,75188.

**Jalur pemakaian** — satu tensor, dua konsumen:

```python
# src/job_runner.py:116-117
cw = json.load(open(meta, encoding="utf-8"))["class_weights"]
weight = torch.tensor([cw["0"], cw["1"]], dtype=torch.float, device=device)
```

- **RM-a:** `crit = nn.CrossEntropyLoss(weight=ctx["weight"])` — `src/tuning.py:132`.
- **RM-b (head):** `crit = nn.CrossEntropyLoss(weight=weight)` — `src/tuning.py:187`, dengan
  `weight` yang diteruskan sebagai `ctx["weight"]` dari `src/job_runner.py:266`. **Objek tensor yang
  sama.**
- **RM-c (agregasi tetangga): TIDAK ada pembobotan kelas.** `retrieval_distribution`
  (`src/rac.py:59-87`) menjumlahkan bobot mentah per kelas — `w = clip(sims, 0, None)` untuk
  `weighting='similarity'` atau `w = 1` untuk `'uniform'` — lalu menormalkan barisnya. Tidak ada
  prior kelas, tidak ada koreksi imbalance.

**Penyeimbangan lain (oversampling / undersampling / SMOTE / `WeightedRandomSampler`): TIDAK ADA.**
Pencarian `smote|oversampl|undersampl|resample|WeightedRandomSampler` pada `src/`, `app.py`,
`run_local_training.py`, dan `scripts/` tidak menghasilkan kecocokan. `DataLoader` memakai
`shuffle=True` biasa (`src/tuning.py:101`, `:189`).

**Vonis: SESUAI.** Penanganan ketidakseimbangan **identik** pada RM-a dan RM-b (bobot yang sama, pada
fungsi loss yang sama, dari sumber yang sama), sehingga perbandingan performa keduanya setara pada
aspek ini. Yang perlu dinyatakan di naskah: pada RM-c, distribusi akhir adalah campuran antara cabang
head yang **berbobot kelas** dan cabang retrieval yang **tidak berbobot kelas** — cabang retrieval
mengikuti prior empiris tetangga (train 81,8% kelas 0), sehingga kenaikan `alpha` secara implisit
menarik prediksi ke arah distribusi mayoritas.

---

### C1 — Satuan `peak_mem_mb`

**Klaim naskah:** RM-a 3.091,8 MB, RM-b 63,9 MB, penghematan 97,93%.

```python
# src/evaluate.py:122-126
def peak_gpu_mem_mb() -> float:
    """Peak GPU memory teralokasi (MB) sejak reset terakhir. 0 bila CPU."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0
```

- **Fungsi:** `torch.cuda.max_memory_allocated()` — memori *allocated*, bukan `max_memory_reserved()`
  (caching allocator). Yang dilaporkan adalah yang benar-benar dipegang tensor, bukan yang dipesan
  dari driver.
- **Pembagi:** `1024 ** 2`. Byte / 1.048.576 = **mebibyte (MiB)**, bukan megabyte desimal.
- **`reset_peak_memory_stats()` dipanggil sebelum tiap pengukuran:** ya — RM-a `src/tuning.py:137-138`
  (setelah model dipindah ke GPU, sehingga bobot model menjadi baseline yang ikut terhitung), RM-b
  `src/tuning.py:190-191` (setelah tensor fitur dipindah ke GPU), ekstraksi fitur
  `src/job_runner.py:140-141`, benchmark inferensi final `src/job_runner.py:543-544`.
- **Cakupan RM-b:** **hanya fase pelatihan head.** Reset di `src/tuning.py:191` terjadi jauh setelah
  ekstraksi fitur selesai dan encoder dihapus (`src/job_runner.py:155-157`). Peak fase ekstraksi
  diukur terpisah dan disimpan sebagai `extract_peak_gpu_mem_mb = 529.5` di
  `results/vast/features/indobenchmark__indobert-base-p2/extract_meta.json`
  (`src/job_runner.py:153`) — nilai itu **tidak pernah** masuk ke kolom `peak_mem_mb` di
  `runs_rmb.csv` maupun ke `best.json`.

**Vonis: TIDAK SESUAI**, pada dua hal.

1. **Satuan.** Angka yang tercatat adalah MiB yang dilabeli MB. Padanan yang benar:
   - RM-a: 3.091,8 MiB = **3.242,0 MB** (≈ 3,24 GB) = 3,02 GiB
   - RM-b: 63,9 MiB = **67,0 MB**
   Persentase penghematan tidak terpengaruh (rasio identik): 97,93% tetap sah.
2. **Cakupan.** 63,9 MiB hanya mengukur pelatihan head. Bila yang dibandingkan adalah "puncak memori
   untuk menghasilkan model terlatih" — sepadan dengan 3.091,8 MiB milik RM-a yang mencakup seluruh
   proses — angka RM-b yang relevan adalah puncak fase ekstraksi, **529,5 MiB (= 555,2 MB)**, yang
   menghasilkan penghematan 82,9%, bukan 97,93%. Naskah perlu menyatakan cakupan mana yang dimaksud.

---

### C2 — Protokol pengukuran inference latency

Semua angka naskah (9,3978 / 9,0789 / 10,9568 ms) berasal dari `phase_final`
(`src/job_runner.py:516-551`) → `results/vast/metrics/inference_benchmark.csv`, lewat harness tunggal:

```python
# src/evaluate.py:129-146
@torch.no_grad()
def measure_latency(predict_fn, sample, n_warmup: int = 5, n_runs: int = 50) -> float:
    cuda = torch.cuda.is_available()
    for _ in range(n_warmup):
        predict_fn(sample)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(sample)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000.0
```

dipanggil identik untuk ketiga skenario: `lat = E.measure_latency(fn, None, n_warmup=10, n_runs=100)`
(`src/job_runner.py:545`), di dalam satu loop `for name, fn in [("RM-a", infer_rma), ("RM-b",
infer_rmb), ("RM-c", infer_rmc)]` (`src/job_runner.py:542`).

| Aspek | RM-a | RM-b | RM-c |
|---|---|---|---|
| Iterasi pemanasan | 10 | 10 | 10 |
| Jumlah pengulangan | 100 | 100 | 100 |
| Statistik dilaporkan | **rata-rata** (total/100) | rata-rata | rata-rata |
| Batch size | 1 (`DataLoader(ds, batch_size=1)`, `src/job_runner.py:519`) | 1 | 1 |
| Tokenisasi dalam timer | **TIDAK** — `ids1/attn1/tti1` ditokenisasi sekali di `src/job_runner.py:517-521`, di luar timer | TIDAK (tensor yang sama) | TIDAK |
| `torch.cuda.synchronize()` | ya, mengapit blok (`src/evaluate.py:140,145`) | ya | ya |
| `eval()` | `model.eval()` (`src/job_runner.py:493`) | `head.eval()` (`:494`) + encoder `eval()` (`src/modeling.py:75`) | sama seperti RM-b |
| `no_grad` | `with torch.no_grad()` (`:525`) + dekorator `@torch.no_grad()` | `:529` | `:534` |

**Isi timer per skenario** (`src/job_runner.py:524-539`):

```python
def infer_rma(_):
    with torch.no_grad(), torch.amp.autocast(...):
        return model(input_ids=ids1, attention_mask=attn1, token_type_ids=tti1).logits

def infer_rmb(_):
    with torch.no_grad(), torch.amp.autocast(...):
        o = encoder(input_ids=ids1, attention_mask=attn1, token_type_ids=tti1)
        return head(M.mean_pool(o.last_hidden_state, attn1).float())

def infer_rmc(_):
    with torch.no_grad(), torch.amp.autocast(...):
        o = encoder(input_ids=ids1, attention_mask=attn1, token_type_ids=tti1)
        pbq = rac.softmax(head(M.mean_pool(o.last_hidden_state, attn1).float()).cpu().numpy())
    s, i = rac.retrieve(index, qe, int(ccfg["k"]))
    pr = rac.retrieval_distribution(s, feats["train"][1][i], 2, weighting=ccfg.get("weighting", "similarity"))
    return rac.fuse(pbq, pr, float(ccfg["alpha"]))
```

**Komponen tambahan RM-c di dalam timer:** transfer GPU→CPU + konversi numpy (`.cpu().numpy()`),
`rac.softmax` di numpy, `rac.retrieve` (yang mencakup `l2_normalize` kueri + `index.search` FAISS
**di CPU**), `retrieval_distribution` (loop numpy atas 2 kelas), dan `fuse`. **Pembangunan indeks
tidak** dalam timer (dilakukan sekali di `src/job_runner.py:497`).

**Vonis: SEBAGIAN.** Harness, jumlah warmup/run, statistik, batch size, mode `eval`, `no_grad`,
autocast, dan penempatan `synchronize` **identik** untuk ketiganya, diukur di satu sesi GPU yang
sama (RTX 3090, `results/vast/hardware.json`) — jadi ketiga angka layak dibandingkan. Tiga perbedaan
protokol yang tetap harus dilaporkan:

1. **Sinkronisasi per-iterasi.** `infer_rmc` memanggil `.cpu().numpy()` **setiap iterasi**, yang
   memaksa sinkronisasi device di tengah pengukuran. `infer_rma`/`infer_rmb` mengembalikan tensor GPU
   tanpa sinkronisasi, sehingga 100 iterasinya diantrekan secara asinkron dan hanya disinkronkan di
   akhir. Sebagian dari selisih +1,88 ms RM-c berasal dari perbedaan pola sinkronisasi ini, bukan
   semata dari biaya retrieval.
2. **Vektor kueri RM-c berasal dari cache, bukan dari encoder yang baru dijalankan.**
   `qe = feats["test"][0][:1]` (`src/job_runner.py:522`) — embedding pra-hitung dari `.npy`; hasil
   `mean_pool` yang dihitung di dalam `infer_rmc` hanya dipakai untuk cabang head. Nilainya sama,
   tetapi jalur inferensi yang diukur melewatkan biaya memindahkan embedding hasil encoder ke
   host untuk dijadikan kueri.
3. **Satu sampel tetap, bukan distribusi.** Yang diulang 100 kali adalah sampel test pertama yang
   sama. Karena `padding="max_length"` (`src/dataset.py:92`) membuat setiap masukan berukuran 128
   token, biaya komputasinya seragam — tetapi angka yang dilaporkan tetap rata-rata satu titik, bukan
   rata-rata atas 1.405 sampel test, dan tidak ada dispersi (std/median/p95) yang tercatat.

Catatan tambahan: kolom `infer_latency_ms` di `results/vast/runs_rmb.csv` (mis. 0,1287 ms untuk run
#28) **bukan** angka yang sama — itu latensi head saja tanpa encoder (`src/tuning.py:214`). Naskah
mengutip angka yang benar (9,0789 dari `inference_benchmark.csv`), tetapi kedua angka itu hidup
berdampingan di `results/vast/` dan mudah tertukar.

---

### C3 — Sumber embedding untuk indeks FAISS

**Klaim naskah:** 6.588 vektor, 768 dimensi, 32 bit, dari representasi token `[CLS]` encoder beku.

- **Split sumber:** **train saja**. `rac.build_faiss_index(feats["train"][0])` —
  `src/tuning.py:242` (evaluasi RM-c) dan `src/job_runner.py:497` (benchmark final). Val/test hanya
  berperan sebagai kueri (`src/tuning.py:244`). Terverifikasi juga di kolom `index_vectors = 6588`
  pada seluruh 67 baris `results/vast/runs_rmc.csv`.
- **Strategi pooling:** **masked mean-pooling**, bukan `[CLS]`:

```python
# src/modeling.py:78-87
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)   # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                    # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                          # (B, 1)
    return summed / counts

# src/modeling.py:110 (di dalam extract_features)
pooled = mean_pool(out.last_hidden_state, attn).float().cpu().numpy()
```

  Tidak ada satu pun pengambilan `last_hidden_state[:, 0]` atau `pooler_output` di seluruh
  repositori.
- **Identik dengan fitur masukan head RM-b:** **ya** — keduanya adalah array `.npy` yang sama.
  `ensure_features` menulis `{split}_emb.npy` sekali (`src/job_runner.py:146-147`) dan
  mengembalikan dict yang dipakai head RM-b (`src/tuning.py:185-186`) maupun indeks RM-c
  (`src/tuning.py:242`) dan cabang head RM-c (`src/tuning.py:241`).
- **Diekstraksi ulang atau pakai ulang artefak RM-b:** **pakai ulang.** `ensure_features` hanya
  mengekstraksi bila berkas `.npy` belum ada (`src/job_runner.py:135`); selebihnya memuat dari disk
  (`src/job_runner.py:158-159`).
- **Tipe data:** **float32.** `extract_features` mengembalikan `.astype(np.float32)`
  (`src/modeling.py:116`) dan `l2_normalize` memaksa `dtype=np.float32` (`src/rac.py:33`).
  Diverifikasi langsung: `train_emb.npy` → shape `(6588, 768)`, dtype `float32`; indeks yang dibangun
  → `IndexFlatIP`, `ntotal=6588`, `d=768`.

**Vonis: TIDAK SESUAI** — hanya pada klaim `[CLS]`. Jumlah vektor (6.588), dimensi (768), presisi
(float32), dan sumber train-only semuanya benar. Untuk sub-pertanyaan "apakah indeks dan head RM-c
bekerja di atas ruang representasi yang sama persis": **YA, sama persis** — keduanya membaca array
`.npy` yang identik, dihasilkan oleh satu kali pemanggilan `extract_features` dengan encoder,
tokenizer, `max_length`, dan strategi pooling yang sama.

---

### C4 — Rumus fusi yang benar-benar diimplementasikan

**Kode fusi apa adanya** (`src/rac.py:96-120`):

```python
def fuse(p_bert: np.ndarray, p_retr: np.ndarray, alpha: float) -> np.ndarray:
    """Fusi distribusi probabilitas: (1-alpha)*p_bert + alpha*p_retr."""
    return (1.0 - alpha) * p_bert + alpha * p_retr


def rac_predict(index, train_labels, query_emb, p_bert, k=5, alpha=0.3, weighting="similarity"):
    sims, idx = retrieve(index, query_emb, k)
    neigh_labels = np.asarray(train_labels)[idx]
    p_retr = retrieval_distribution(sims, neigh_labels, num_labels=p_bert.shape[1], weighting=weighting)
    p_final = fuse(p_bert, p_retr, alpha)
    return p_final.argmax(axis=1), p_final
```

- **Softmax pada keluaran head sebelum fusi: YA.** `pb = rac.softmax(head(...).cpu().numpy())` —
  `src/tuning.py:241` (evaluasi RM-c) dan `src/job_runner.py:504`/`:536` (fase final). Implementasi
  softmax numerik-stabil di `src/rac.py:90-93`.
- **Softmax lagi setelah fusi: TIDAK.** `fuse` (`src/rac.py:98`) langsung diikuti
  `p_final.argmax(axis=1)` (`src/rac.py:120`). Tidak ada pemanggilan `softmax` di antara keduanya di
  seluruh repositori.
- **Pembentukan distribusi tetangga** (`src/rac.py:59-87`): bobot per tetangga ditentukan skema
  `weighting` — `'uniform'` → `w = 1` untuk semua tetangga; `'similarity'` → `w = np.clip(sims, 0.0,
  None)`, yaitu cosine similarity dengan nilai negatif dipangkas ke 0. Massa lalu dijumlahkan per
  kelas (`dist[:, c] = (w * mask).sum(axis=1)`, `src/rac.py:77-79`) dan dinormalkan per baris
  (`return dist / total`, `src/rac.py:87`), sehingga tiap baris `p_retr` berjumlah 1.
- **Mekanisme pengaman saat semua similaritas tidak positif: ADA** — `src/rac.py:81-86`:

```python
total = dist.sum(axis=1, keepdims=True)
zero = (total.squeeze(-1) == 0)
if zero.any():
    dist[zero] = 1.0 / num_labels     # jatuh ke distribusi seragam
    total[zero] = 1.0
```

  Pada data ini fallback tersebut **tidak pernah aktif**: similaritas minimum yang terukur atas
  seluruh kueri test (k=5) adalah 0,4161 (0% nilai ≤ 0).

**Vonis: SESUAI.** Fusi memang di level probabilitas, softmax hanya sekali (cabang BERT, sebelum
fusi), dan kelas prediksi ditentukan langsung via `argmax` tanpa normalisasi ulang. Karena `p_bert`
dan `p_retr` sama-sama distribusi dan bobotnya berjumlah 1, `p_final` sudah merupakan distribusi
yang sah — kolom `p_judi` yang dipakai untuk PR curve (`src/tuning.py:249`) adalah `p_final[:, 1]`
yang sama.

---

### C5 — Konfigurasi final RM-c

**Yang tersimpan di artefak** (dibaca langsung, bukan disimpulkan):

| Artefak | Isi |
|---|---|
| `results/vast/checkpoints/rmc_best.pt` | `config = {"alpha": 0.2, "k": 5, "weighting": "similarity"}`, `run_id = 13`, `val_f1_macro = 0.969903` |
| `results/vast/best.json` → `rmc` | `config = {"alpha": 0.2, "k": 5, "weighting": "similarity"}`, `run_id = 13`, `val_f1_macro = 0.969903` |
| `results/vast/metrics/final_comparison.csv` → baris RM-c | `config = {"alpha": 0.2, "k": 5, "weighting": "similarity"}`, `val_f1_macro = 0.969903`, `test_f1_macro = 0.949693` |
| `results/vast/job.log` (fase final) | baris perbandingan mencantumkan `{"alpha": 0.2, "k": 5, "weighting": "similarity"}` |

**Konfigurasi yang benar-benar dipakai saat menghitung metrik test** diambil dari **checkpoint**,
bukan dari `best.json`:

```python
# src/job_runner.py:496
ccfg = torch.load(out / "checkpoints" / "rmc_best.pt", map_location=device)["config"]
...
# src/job_runner.py:510
tm_c, ex_c = T.eval_rmc(ccfg, feats, head, device, split="test")
```

dan `ccfg` yang sama dipakai untuk benchmark latency (`src/job_runner.py:537-539`). Jadi angka test
RM-c di naskah (F1-macro 0,9497, latency 10,9568 ms) **memang dihasilkan dengan alpha 0,2 / k 5 /
weighting similarity** — sesuai naskah.

**Ketidakcocokan nomor run.** Di `results/vast/runs_rmc.csv` yang ada sekarang:

| run_id | alpha | k | weighting | val_f1_macro |
|---|---|---|---|---|
| 13 | 0.2 | **1** | similarity | 0.969995 |
| 15 | 0.2 | **5** | similarity | **0.969903** |

Nilai `val_f1_macro = 0.969903` dan konfigurasi `k=5` yang tersimpan di `best.json`/`rmc_best.pt`
cocok persis dengan **baris run #15**, sementara field `run_id`-nya tertulis **13** — dan baris run
#13 pada CSV justru `k=1` dengan F1 0,969995. Ini tidak bisa dihasilkan oleh alur kode normal:
`update_best` (`src/job_runner.py:95-105`) dan penyimpanan checkpoint (`src/job_runner.py:370`)
selalu menulis `run_id`, `config`, dan `val_f1_macro` dari run yang sama, dan run #15 (F1 lebih
rendah dari juara saat itu) tidak akan lolos syarat `payload["val_f1_macro"] > prev[...]`
(`src/job_runner.py:100`).

**Kesimpulan berbasis nilai (sesuai instruksi: cocokkan berdasarkan konfigurasi, bukan nomor run):**
artefak model dan indeks final beroperasi pada **alpha = 0,2, k = 5, weighting = similarity**, di
atas head RM-b run #28 (`head_arch=mlp`, `hidden_dim=1024`, `epochs=10`, `lr=1e-3`, `dropout=0.1`,
`weight_decay=0.0`, `batch=32`) yang tersimpan di `checkpoints/rmb_best.pt`. Indeks FAISS-nya dibangun
ulang saat itu juga dari `feats["train"]` (6.588 vektor) — tidak ada artefak indeks tersimpan di
`results/vast/`.

**Ketidakcocokan ini sudah terdokumentasi di dalam repositori.** `scripts/merge_runs.py:85-93`
memuat catatan eksplisit: `run_id` di `best.json` bergeser sementara config + F1-nya benar, dan run
#13 (`k=1`) sengaja **tidak** dipilih meski 0,009 pp lebih tinggi karena `k=1` dinilai tidak stabil —
alasan yang sama tercatat di `run_candidate2_vast.sh:76` dan di kolom `catatan` run #67
`runs_rmc.csv`. Jadi pemilihan k=5 adalah keputusan manusia yang terdokumentasi, bukan keluaran
mekanis `update_best`.

**Vonis: SEBAGIAN.** Konfigurasi yang dilaporkan naskah **benar** dan cocok dengan artefak serta
dengan angka test yang dilaporkan. Yang tidak konsisten adalah **penomoran run**: label `run_id: 13`
di `best.json`/`rmc_best.pt` tidak menunjuk baris yang benar pada `runs_rmc.csv` saat ini
(konfigurasi itu ada di baris run #15). Naskah sebaiknya merujuk konfigurasi (α=0,2; k=5;
similarity), bukan nomor run. Konsekuensi lain yang perlu disadari saat menulis: karena juara
mekanis menurut val F1 adalah k=1 (0,969995) dan yang dipakai adalah k=5 (0,969903), naskah tidak
boleh menyatakan konfigurasi final RM-c sebagai "hasil pemilihan otomatis berdasarkan val F1-macro
tertinggi" — itu hasil penilaian stabilitas oleh peneliti di atas dua sel yang praktis seri.

---

### C6 — Disiplin data uji

**Klaim naskah:** data uji tidak digunakan pada tahap penalaan.

Bukti dari sisi kode:

- **Default OFF.** `eval_test` hanya aktif bila `job["eval_test"]` bernilai true
  (`src/job_runner.py:258`, `:327`, `:383`; `bool(job.get("eval_test"))`). Di UI, ketiga skenario
  memakai `st.checkbox(...)` tanpa `value=True` → default `False` (`app.py:313`, `:335`, `:364`).
  Batch mode juga default `False` (`app.py:418` menulis `"eval_test": False`; `app.py:491`
  memakai `False` bila kolom tidak ada).
- **Seleksi memakai validasi saja.** `update_best` membandingkan `payload["val_f1_macro"]`
  (`src/job_runner.py:100`); `RunLogger.log` menghitung delta terhadap `val_f1_macro`
  (`src/tuning.py:81-86`); pemilihan epoch terbaik memakai `vm["f1_macro"]` dari **val**
  (`src/tuning.py:161`, `:208`). RM-c dievaluasi dengan `split="val"` (`src/job_runner.py:347`).
- **Tidak ada early stopping berbasis test.** Tidak ada early stopping sama sekali — seluruh epoch
  dijalankan, hanya bobot terbaik-menurut-val yang disimpan (`src/tuning.py:161-164`, `:208-210`).

Bukti dari sisi artefak (kampanye yang benar-benar berjalan):

| Berkas | Baris | Kolom `test_*` |
|---|---|---|
| `results/vast/runs_rma.csv` | 26 | **tidak ada** |
| `results/vast/runs_rmb.csv` | 31 | **tidak ada** |
| `results/vast/runs_rmc.csv` | 67 | **tidak ada** |

Karena kolom `test_*` hanya ditulis ketika `eval_test` true (`src/job_runner.py:219-228`, `:278-287`,
`:356-365`), ketiadaan kolom itu di seluruh 124 baris membuktikan **tidak satu pun run penalaan
mengevaluasi test**. Seluruh CSV grid di `tuning_grids/` juga tidak memiliki kolom `eval_test`
(header terverifikasi untuk 8 berkas), sehingga batch mana pun berjalan dengan `eval_test=False`.
Test baru dibuka di `phase_final` (`src/job_runner.py:500-514`), yang dijalankan dari `job.json`
terakhir (`{"phase": "final", "out_dir": "results/vast", "smoke": false}`).

**Vonis: SESUAI — TIDAK ada penggunaan data uji pada penalaan** (baik untuk evaluasi, pemilihan
model, maupun early stopping). Dua catatan kejujuran yang layak masuk naskah:

1. **Teks test melewati encoder saat ekstraksi fitur** (`src/job_runner.py:143`, ketiga split
   diekstraksi sekaligus). Yang dihasilkan hanyalah embedding dari encoder **beku** yang tidak
   pernah dilatih; label test tidak pernah dibaca dan tidak ada metrik test yang dihitung. Ini bukan
   kebocoran ke arah pemilihan hyperparameter, tetapi cakupannya perlu disebut karena
   `extract_time_s = 7,90 s` yang masuk ke `train_time_s` RM-b mencakup pekerjaan itu (lihat A1).
2. `run_local_training.py:184-197` memilih `k`/`alpha` RM-c pada val lalu mengevaluasi test dalam
   satu skrip. Skrip ini menulis ke `results/local/`, dan folder tersebut **kosong** — jadi bukan
   sumber angka mana pun di Bab 4.
3. **Di luar `results/vast/`, test memang pernah dibuka — semuanya bertanggal setelah benchmark
   final 2026-07-25:**
   - `results_c2/vast_candidate2/` — 4 run (`runs_rma_c2.csv` 1, `runs_rmb_c2.csv` 2,
     `runs_rmc_c2.csv` 1) dengan kolom `test_*` terisi; `run_candidate2_vast.sh:32,47,77` memang
     menyetel `"eval_test": true`. Ini eksplorasi kandidat #2 (α=0,2/k=3), bukan penalaan.
   - `results/vast_rmc_cheap_head/runs_rmc_cheap.csv` — 69 baris (2026-08-03), tetapi hanya **3**
     yang punya nilai `test_f1_macro`; 66 baris grid α×k-nya berjalan val-only, persis seperti
     kampanye resmi.
   Tidak satu pun folder ini menjadi sumber angka Bab 4 (`CLAUDE.md` mengunci `results/vast/`), dan
   tidak satu pun mendahului penetapan konfigurasi final. Namun bila naskah menuliskan "test baru
   dibuka satu kali", kalimat itu perlu dibatasi pada kampanye `results/vast/`.

---

## 3. Temuan Prioritas

Vonis **TIDAK SESUAI** dan dampaknya terhadap klaim naskah:

### P1 — Representasi RM-c bukan token `[CLS]`, melainkan mean-pooling (C3)

Kode memakai masked mean-pooling atas seluruh token non-padding (`src/modeling.py:78-87`, dipakai di
`src/modeling.py:110`). Tidak ada `[CLS]` di mana pun.
**Klaim yang terdampak:** deskripsi indeks FAISS di Bab 3/Bab 4 ("dibangun dari representasi token
`[CLS]` hasil encoder beku"), dan — karena fitur yang sama menjadi masukan head — deskripsi RM-b
juga. Ini kesalahan deskripsi metode, bukan kesalahan hasil: seluruh angka tetap valid, tetapi
metodenya salah dinarasikan. Perlu diperbaiki di **dua** tempat sekaligus (RM-b dan RM-c) supaya
tetap konsisten dengan pernyataan "indeks dan head bekerja di ruang representasi yang sama".

### P2 — `peak_mem_mb` adalah MiB yang dilabeli MB, dan cakupan RM-b tidak sepadan (C1)

`torch.cuda.max_memory_allocated() / (1024 ** 2)` (`src/evaluate.py:125`) menghasilkan MiB.
**Klaim yang terdampak:** "RM-a 3.091,8 MB" seharusnya 3.091,8 MiB = 3.242,0 MB; "RM-b 63,9 MB"
seharusnya 63,9 MiB = 67,0 MB. Persentase 97,93% tidak terdampak.
Selain itu 63,9 MiB **hanya** mencakup pelatihan head; puncak fase ekstraksi fitur (529,5 MiB,
`extract_meta.json`) tidak diikutkan. Bila naskah membandingkan "puncak memori untuk memperoleh
model terlatih", angka pembandingnya menjadi 529,5 MiB dan penghematannya 82,9%, bukan 97,93%.
Naskah harus menyatakan cakupan yang dipilih.

### P3 — Bab 3 mengklaim penyaringan yang tidak ada di kode (B4)

Tidak ada filter panjang minimum, deteksi bahasa, maupun pembuangan komentar non-teks.
**Klaim yang terdampak:** kalimat Bab 3 "komentar dengan panjang teks tidak memadai serta komentar
non-teks dieliminasi". Bab 4 (tiga tahap: baris kosong, dedup, guard) sudah benar, sehingga Bab 3 dan
Bab 4 saat ini **saling bertentangan** — dan yang keliru adalah Bab 3.

### Catatan derajat dua (vonis SEBAGIAN, tidak membatalkan angka tetapi mengubah cara menuliskannya)

- **A1** — 81,10 s dan 11,65 s tidak mengukur cakupan yang identik: RM-a memuat 5× evaluasi validasi
  ber-encoder penuh dan penyalinan 109,5 juta parameter; RM-b memuat ekstraksi **tiga** split
  (termasuk test) yang diukur sekali lalu dipakai ulang di 31 run. Angka 85,64% tetap boleh dikutip,
  tetapi harus disertai definisi cakupan kedua sisi.
- **B3** — penyajian "empat tahap berurutan" menyembunyikan posisi split dan `clean_text`, serta
  sifat guard sebagai operasi pasca-split pada `text_clean` (bukan penyaringan korpus). 9.395 bukan
  jumlah baris unik hasil dedup.
- **C2** — protokol harness identik, tetapi RM-c menanggung sinkronisasi device per-iterasi dan
  memakai vektor kueri dari cache; selisih +1,88 ms tidak seluruhnya biaya retrieval.
- **C5** — konfigurasi final (α=0,2; k=5; similarity) benar dan cocok dengan artefak, tetapi
  `run_id: 13` di `best.json`/`rmc_best.pt` tidak menunjuk baris yang benar di `runs_rmc.csv`
  (konfigurasi itu adalah run #15). Rujuk konfigurasi, jangan nomor run.

---

## 4. Tidak Dapat Diverifikasi

Hal-hal yang tidak bisa dipastikan dari kode + artefak yang ada di repositori:

1. **Porsi 81,10 s RM-a yang habis untuk evaluasi validasi per epoch.** `history/rma_history.csv`
   hanya menyimpan metrik per epoch, tanpa kolom waktu; `train_eval_rma` tidak mencatat waktu
   per-fase (bandingkan `run_local_training.py:87,98` yang mencatat `epoch_time_s` — tetapi skrip itu
   tidak pernah dijalankan, `results/local/` kosong). Memisahkan komponennya menuntut menjalankan
   ulang training di GPU, yang di luar cakupan audit ini.

2. **Apakah `extract_time_s = 7,9 s` diukur pada sesi GPU yang sama dengan `train_time_s = 81,1 s`
   RM-a.** `extract_meta.json` mencatat `extracted_at = 2026-07-25 09:31:00` dan run RM-b #28
   ber-timestamp `2026-07-25 09:48:36`, sementara `hardware.json` (RTX 3090) ditulis ulang setiap
   job. Tanggal dan jamnya berdekatan dan konsisten dengan satu sesi, tetapi kode tidak menyimpan
   ID sesi/instance, sehingga "satu sesi" tidak dapat dibuktikan dari artefak — hanya sangat mungkin.

3. **Riwayat yang menghasilkan ketidakcocokan `run_id` RM-c (C5).** Yang bisa dipastikan hanya
   keadaan akhir: `best.json`/`rmc_best.pt` memuat konfigurasi k=5 dengan label run #13. Penjelasan
   yang konsisten dengan kode adalah adanya kampanye RM-c **sebelumnya** dengan grid berukuran lain
   (mis. 5 nilai k, sehingga sel α=0,2/k=5 jatuh di urutan ke-13) yang menulis `best.json` dan
   checkpoint, lalu `runs_rmc.csv` di-reset dan grid dijalankan ulang dengan 6 nilai k
   (`tuning_grids/RMC_TUNING_GRID.csv`, 66 baris) sehingga penomoran bergeser — `RunLogger.next_id()`
   memang menurunkan id dari jumlah baris CSV (`src/tuning.py:71-72`), dan `src/job_runner.py:180-181`
   mendokumentasikan bahwa id bisa terulang bila CSV pernah di-reset. Namun `results/vast/` tidak
   ter-*version control* (`.gitignore:38`) dan tidak ada log run terdahulu, sehingga rekonstruksi ini
   **hipotesis, bukan fakta terverifikasi**.

4. **Apakah `results/vast/best.json` pernah disunting manual.** Isi `rmc` di dalamnya tidak bisa
   dihasilkan oleh `update_best` untuk run #13 pada CSV yang ada sekarang, tetapi tidak ada jejak
   audit (mtime saja tidak cukup, berkas tidak ter-track git) yang membedakan "disunting tangan" dari
   "sisa kampanye sebelumnya". `scripts/merge_runs.py:85-93` mencatat pergeseran itu sebagai fakta
   yang diketahui dan sengaja tidak dikoreksi, tanpa menyebut penyebabnya.

5. **Klaim naskah Bab 3 itu sendiri.** Naskah Bab 3 dan Bab 4 tidak ada di repositori; `draft_bab4/`
   berisi delapan berkas draf tetapi tidak memuat kata "CLS" maupun pembahasan normalisasi
   slang/kata tidak baku. Seluruh audit ini membandingkan kode terhadap klaim **sebagaimana
   dirumuskan dalam perintah audit**, bukan terhadap teks naskah yang final.

6. **Reproduksi angka baris 14.237 → 14.227 → 9.412.** Angka-angka itu dibaca dari
   `dataset/processed/metadata.json` dan dari `assert` di dalam notebook
   (`assert n_dedup == 9412`, cell 5), bukan dari eksekusi ulang notebook — menjalankan ulang
   `02_preprocessing.ipynb` akan menimpa `dataset/splits/`, yang dilarang oleh sifat read-only audit
   ini.

7. **Perilaku `weighting='uniform'` pada konfigurasi final.** Tidak relevan untuk angka Bab 4
   (final memakai `similarity`), tetapi `runs_rmc.csv` run #67 mencatat `val_f1_macro` yang identik
   untuk `uniform` dan `similarity` di sel α=0,2/k=5 (0,969903) — kode tidak menyimpan prediksi
   per-sampel, sehingga tidak bisa dipastikan apakah kedua skema menghasilkan prediksi yang sama
   persis atau kebetulan bermetrik sama.
