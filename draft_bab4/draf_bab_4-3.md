# 4.3 Hasil Eksplorasi Hyperparameter (Tuning)

Subbab ini menyajikan hasil eksplorasi hyperparameter ketiga skenario adaptasi — RM-a
(*full fine-tuning*), RM-b (*frozen encoder*), dan RM-c (*frozen encoder* + RAC). Untuk
menjaga cakupan tabel dan grafik tetap ringkas, hanya **2 konfigurasi teratas dan 2
konfigurasi terbawah** (berdasarkan `val_f1_macro`) yang disajikan per skenario; daftar
lengkap seluruh konfigurasi yang dicoba tersedia pada `results/vast/runs_{rma,rmb,rmc}.csv`.

## 4.3.1 RM-a (Full Fine-tuning)

Eksplorasi hyperparameter RM-a menggunakan metode **hybrid**: grid kombinatorial untuk tiga
*hyperparameter* yang saling terkopel (`lr`, `epochs`, `batch`) — karena ketiganya bersama
menentukan lintasan optimisasi (jumlah *update* optimizer = `⌈N/batch⌉ × epochs`) — dilanjutkan
*coordinate descent* untuk dua *hyperparameter* yang relatif independen (`warmup_ratio`,
`weight_decay`). Total **26 konfigurasi** dieksplorasi: 24 titik grid kombinatorial
(`lr` ∈ {1e-5, 2e-5, 3e-5, 5e-5} × `epochs` ∈ {3, 5, 8} × `batch` ∈ {16, 32}) ditambah 2 titik
*coordinate descent* pada sel pemenang.

Konfigurasi #1 yang diuji adalah baseline kanonik dari literatur IndoNLU/Wilie dkk. (2020)
(`lr`=2e-5, `epochs`=5, `batch`=32, `warmup_ratio`=0,1, `weight_decay`=0,01), dijalankan
lebih dahulu sebagai *sanity check* sekaligus titik acuan seluruh grid. Hasil akhir
menunjukkan baseline ini **tetap menjadi konfigurasi terbaik** di seluruh 26 konfigurasi yang
diuji — *coordinate descent* pada `weight_decay` (diuji 0,10 sebagai alternatif 0,01)
menghasilkan `val_f1_macro` yang identik hingga lima desimal, menunjukkan model RM-a tidak
sensitif terhadap regularisasi *weight decay* pada rentang yang diuji.

| Peringkat | Konfigurasi | val_f1_macro |
|---|---|---|
| Atas #1 | `lr`=2e-5, `epochs`=5, `batch`=32, `wd`=0,01 | 0,98216 |
| Atas #2 | `lr`=2e-5, `epochs`=5, `batch`=32, `wd`=0,10 (seri dengan #1) | 0,98216 |
| Bawah #2 | `lr`=1e-5, `epochs`=3, `batch`=16, `wd`=0,01 | 0,96808 |
| Bawah #1 | `lr`=5e-5, `epochs`=3, `batch`=16, `wd`=0,01 | 0,96705 |

Tabel 4.6 Dua konfigurasi teratas dan terbawah RM-a (`results/vast/runs_rma.csv`)

Perlu dicatat bahwa **rentang nilai `val_f1_macro` di seluruh grid RM-a relatif sempit**
(0,96705 – 0,98216, selisih ±1,5pp antara konfigurasi terbaik dan terburuk) — mengindikasikan
*full fine-tuning* IndoBERT pada tugas ini secara umum robust terhadap variasi hyperparameter
dalam rentang yang wajar, dan konfigurasi kanonik dari literatur sudah mendekati optimal untuk
dataset ini. Konfigurasi terbaik (Atas #1) dipilih sebagai konfigurasi final RM-a dan
digunakan pada evaluasi *test set* (§4.4).

## 4.3.2 RM-b (Frozen Encoder)

Eksplorasi hyperparameter RM-b disusun bertahap mengikuti struktur ketergantungan: keputusan
**arsitektur head** (linear vs. MLP beserta `hidden_dim`) diuji lebih dahulu karena bersifat
struktural (`hidden_dim` hanya relevan untuk arsitektur MLP), disusul grid `lr` × `epochs`
pada arsitektur pemenang, dan ditutup *coordinate descent* pada `dropout`/`weight_decay`/
`batch`. Total **27 konfigurasi** dieksplorasi: 4 titik tahap arsitektur + 2 titik perluasan
(karena `hidden_dim`=512 sempat menang di tepi rentang yang diuji) + 15 titik grid
`lr` × `epochs` + 6 titik *coordinate descent*.

Temuan paling menonjol dari tahap arsitektur adalah **lompatan performa besar dari head
*linear* ke MLP**, jauh melampaui variasi yang dihasilkan tuning hyperparameter di dalam satu
arsitektur:

| Arsitektur | Trainable params | val_f1_macro | Kenaikan marjinal (pp F1 / 100rb param) |
|---|---|---|---|
| Linear (baseline) | 1.538 | 0,9125 | — |
| MLP, `hidden_dim`=128 | 98.690 | 0,9368 | **2,51** (lompatan terbesar) |
| MLP, `hidden_dim`=256 | 197.378 | 0,9400 | 0,32 |
| MLP, `hidden_dim`=512 | 394.754 | 0,9487 | 0,44 |
| MLP, `hidden_dim`=1024 | 789.506 | 0,9543 *(tahap arsitektur, sebelum tuning lr/epochs)* | 0,16–0,12 |

Tabel 4.7 Efisiensi marjinal kenaikan `hidden_dim` terhadap `val_f1_macro` (`results/vast/runs_rmb.csv`)

Pola ini menunjukkan bahwa **hampir seluruh manfaat menambah kapasitas head sudah diperoleh
pada lompatan pertama** (linear → MLP kecil); penambahan kapasitas lebih lanjut memberi
kenaikan performa dengan laju yang jauh menurun (*diminishing returns*). Setelah arsitektur
`hidden_dim`=1024 dikunci, grid `lr` × `epochs` dan *coordinate descent* selanjutnya
menghasilkan konfigurasi final `lr`=1e-3, `epochs`=10, `dropout`=0,1, `weight_decay`=0,0,
`batch`=32.

| Peringkat | Konfigurasi | val_f1_macro |
|---|---|---|
| Atas #1 | MLP/1024, `epochs`=10, `lr`=1e-3, `wd`=0,0 | 0,96530 |
| Atas #2 | MLP/1024, `epochs`=10/20/30, `lr`=1e-3, `wd`=0,01 (seri) | 0,96498 |
| Bawah #1/#2 | Linear (baseline kanonik) | 0,91251 |

Tabel 4.8 Dua konfigurasi teratas dan terbawah RM-b (`results/vast/runs_rmb.csv`)

*Catatan: `runs_rmb.csv` mentah memuat 31 baris karena 4 baris tahap arsitektur sempat
ter-*upload* dobel saat kampanye berlangsung; 27 adalah jumlah konfigurasi rancangan yang
distinct.*

## 4.3.3 RM-c (Frozen Encoder + RAC)

RM-c mengevaluasi fusi probabilitas antara head RM-b dan distribusi hasil retrieval
*k*-nearest neighbor berbasis indeks FAISS (dibangun hanya dari embedding *train set*,
menjaga prinsip anti-*leakage*), tanpa proses pelatihan tambahan:

$$p_{final} = (1-\alpha) \cdot \text{softmax}(head(x)) + \alpha \cdot p_{retrieval}$$

dengan $\alpha$ mengatur bobot cabang retrieval terhadap cabang head, dan $k$ menentukan
jumlah tetangga yang dipertimbangkan.

Karena RM-c tidak melibatkan pelatihan ulang, rancangan grid $\alpha \times k$
($\alpha \in \{0{,}0, 0{,}1, \dots, 1{,}0\}$, 11 titik; $k \in \{1,3,5,10,20,50\}$, 6 titik;
*weighting*=*similarity* dikunci; 67 konfigurasi per kondisi termasuk 1 cek
*weighting*=*uniform*) dijalankan pada **dua kondisi head RM-b sekaligus**, untuk menguji
kontribusi RAC terhadap head dengan kapasitas berbeda secara langsung dalam satu
rancangan eksperimen:

1. **Head performa-terbaik** dari hasil §4.3.2 (arsitektur MLP, `hidden_dim`=1024,
   789.506 parameter)
2. **Head paling ringan** dari hasil §4.3.2 (arsitektur *linear*, 1.538 parameter —
   513× lebih kecil)

### Hasil grid — head performa-terbaik

Pemenang **mekanis** tertinggi berada di $\alpha=0{,}2, k=1$ (val F1-macro 0,9700), namun
konfigurasi $k=1$ menunjukkan pola tidak stabil di sepanjang sumbu $\alpha$ (estimator
varians tinggi, hanya bergantung satu tetangga). Konfigurasi $\alpha=0{,}2, k=5$
(val F1-macro 0,9699) seri secara statistik (selisih 0,01pp, di bawah ambang 0,15pp)
dengan kurva jauh lebih stabil, sehingga dipilih sebagai konfigurasi final.

| Parameter | Nilai final |
|---|---|
| $\alpha$ | 0,2 |
| $k$ | 5 |
| *weighting* | *similarity* |

**2 teratas + 2 terbawah:**

| Peringkat | Config | val_f1_macro |
|---|---|---|
| Atas #1 | α=0,2, k=1 (mekanis tertinggi, tak stabil) | 0,96999 |
| Atas #2 | α=0,2, k=5 (dipilih final) | 0,96990 |
| Bawah #2 | α=0,9, k=50 | 0,92191 |
| Bawah #1 | α=1,0, k=50 (retrieval murni) | 0,90752 |

### Hasil grid — head paling ringan

Pada kondisi ini, pemenang mekanis ($\alpha=0{,}5, k=3$, val F1-macro 0,9576) sudah
merupakan konfigurasi yang stabil (bukan $k=1$), sehingga langsung ditetapkan sebagai
konfigurasi final tanpa perlu penyesuaian lanjutan.

| Parameter | Nilai final |
|---|---|
| $\alpha$ | 0,5 |
| $k$ | 3 |
| *weighting* | *similarity* |

**2 teratas + 2 terbawah:**

| Peringkat | Config | val_f1_macro |
|---|---|---|
| Atas #1 | α=0,5, k=3 | 0,95760 |
| Atas #2 | α=0,4, k=5 | 0,95712 |
| Bawah #2 | α=0,0, k=1 (head murni, tanpa RAC) | 0,91251 |
| Bawah #1 | α=1,0, k=50 (retrieval murni) | 0,90752 |

### Perbandingan kontribusi RAC terhadap kedua kondisi head

| | Head murni (α=0) | RM-c (α optimal) | Δ val | test F1-macro | Trainable params (head) |
|---|---|---|---|---|---|
| Head performa-terbaik | 0,9653 | 0,9699 | +0,46pp | 0,9497 | 789.506 |
| **Head paling ringan** | 0,9125 | **0,9576** | **+4,51pp** | **0,9513** | **1.538** |

### Pembahasan

Kontribusi RAC terhadap kedua kondisi head menunjukkan pola yang berbanding terbalik
dengan kapasitas head: pada head performa-terbaik, RAC hanya memberi perbaikan tipis
(+0,46pp val), sedangkan pada head paling ringan, RAC menutup gap performa sebesar
4,51pp val — jauh lebih besar. Pada data uji, kondisi head-ringan+RAC (F1-macro 0,9513)
bahkan sedikit melampaui kondisi head performa-terbaik+RAC (F1-macro 0,9497), meskipun
parameter head yang dipakai 513× lebih sedikit. Temuan ini mengindikasikan bahwa
mekanisme retrieval pada RAC dapat mengompensasi keterbatasan kapasitas head secara
efektif, membuka peluang efisiensi lebih lanjut tanpa mengorbankan performa klasifikasi
secara berarti.

**Sumber:** `tuning_grids/RMC_TUNING_GRID.md`, `results/vast/runs_rmc.csv`,
`results/vast_rmc_cheap_head/runs_rmc.csv`.
