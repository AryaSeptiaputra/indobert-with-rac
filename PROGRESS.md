# PROGRESS — IndoBERT-with-RAC

Status proyek skripsi ITENAS 2026 (trade-off performa versus efisiensi tiga
strategi adaptasi IndoBERT untuk deteksi komentar judi). Terakhir diperbarui:
**2026-09-30**, branch `refactor/notebook-style`.

## Ringkasan tahap

| Tahap | Status | Artefak |
|-------|--------|---------|
| EDA | Selesai (ditambah pencarian entitas HTML, 2026-09-30) | `notebooks/01_eda.ipynb`, `docs/EDA_REPORT_BAGIAN1/2/3.md` |
| Preprocessing | Selesai, split dibangun ulang 2026-09-30 | `data/processed/`, `DATASET.md` |
| Refactor ke gaya notebook-only | Selesai (2026-09-30) | `notebooks/`; `src/` dan `tests/` dihapus |
| Kampanye tuning RM-a / RM-b / RM-c | Selesai (RTX 4090, 2026-09-30) | `outputs/tuning/runs_*.csv`, `best.json`, `candidates.json` |
| Benchmark final (test, satu sesi) | Selesai (2026-09-30) | `outputs/tuning/metrics/` |
| Artefak Bab 4 (gambar, data tabel) | Selesai (2026-09-30) | `outputs/tuning/artifacts/` |
| Penulisan Bab 4 | Belum | angka resmi di bagian "Hasil kampanye RTX 4090" |

## Keadaan sekarang

Kampanye ulang di satu instance Vast.ai **RTX 4090 24 GB** selesai pada 2026-09-30:
tuning (03a-03c), benchmark final (05), artefak (06), dan arsip (07) berjalan dalam
satu sesi mesin. Seluruh `outputs/` (termasuk checkpoint dan fitur beku, 1,8 GB) dan
notebook hasil eksekusi (`outputs/notebooks_eksekusi/`) tersimpan di mesin lokal,
di luar git; cadangannya di `C:\Users\Arya\Downloads\HASIL-300926\`. Checkpoint
juara ada di `models/`. Angka kampanye RTX 3090 dan RTX 3050 sebelumnya tidak berlaku
untuk Bab 4, dan arsip RTX 3090 lokal sudah dihapus.

## Hasil kampanye RTX 4090 (2026-09-30)

**Lingkungan** (`hardware.json`): NVIDIA GeForce RTX 4090 24 GB, driver 580.178.04,
CUDA 13.0, AMD Ryzen Threadripper PRO 3975WX (64 thread), RAM 251,5 GB, Ubuntu (Linux
6.8), Python 3.13.15, torch 2.12.1+cu130, transformers 5.12.1, faiss 1.14.3, seed 42.
Sesi tuning dan benchmark final berada pada boot yang sama (`same_boot_as_tuning: true`).

**Jalur tuning (seleksi pada validation):**

- **RM-a** (26 run): grid tahap 1 `lr x epochs x batch` (24 run) memberi permukaan
  datar (rentang 0,9 pp, rata-rata per lr 97,30 ± 0,01). #9 (3e-5 / 8 / 16) 0,97856
  seri dengan #1 (2e-5 / 5 / 16) 0,97713; pusat tahap 2 = juara F1 mentah (#9).
  Tahap 2: warmup 0,0 (-0,39 pp) dan wd 0,1 (-0,27 pp) kalah, default dipertahankan.
  **Juara #9**, epoch terbaik 6 dari 8, 136,1 s, puncak memori latih 3.599 MB.
- **RM-b** (31 run): tahap 1 mlp/512 menang di tepi (0,9470) -> tahap 1B 768/1024
  seri tiga arah, dikunci mlp/512 (termurah). Tahap 2 `lr x epochs`: lr 2e-4 / 30
  epoch menang di tepi epoch -> tahap 2B 50/100 epoch (jenuh, puncak epoch 66). Tahap
  3 di #23: hanya dropout 0,3 berpengaruh (+0,44 pp) -> tahap 3B 0,4/0,5 memastikan
  0,3 adalah puncak. **Juara #25** (mlp/512, lr 2e-4, 100 epoch, dropout 0,3, wd 0,01,
  batch 32) 0,968456, 394.754 parameter, 21,6 s (ekstraksi 3,9 s + head 17,8 s),
  puncak memori latih 529,5 MB.
- **RM-c** (2.047 run): 31 head x 66 `alpha x k` + cek uniform. Gain RAC berbanding
  terbalik dengan kekuatan head (+4,25 pp pada head linear, di bawah ambang seri pada
  head >= 0,96); kurva alpha rata-rata memuncak di 0,4 dan negatif di atas sekitar
  0,75. Default (head #25, alpha 0,1, k=1) 0,969622 (+0,12 pp atas head sendiri);
  penantang head #30 alpha 0,3 k=1 +0,26 pp tetapi CI95 bootstrap [-0,59; +1,10] pp
  melewati nol -> **juara default #1591**, biaya = head RM-b resmi.

**Kandidat** (`candidates.json`, sha256 `72430138…a04`, ditetapkan 09:02:11 sebelum
test dibuka): RM-a #9 / #1, RM-b #25 / #30, RM-c #1591 / #1597 (alpha 0,2).

**Benchmark final (test, 1.405 sampel):**

| | RM-a | RM-b | RM-c |
|---|---|---|---|
| F1-macro | 0,963478 | 0,949227 | 0,949227 |
| F1 judi (precision / recall) | 0,9405 (0,925 / 0,957) | 0,9167 (0,931 / 0,902) | sama dengan RM-b |
| Selisih F1 vs RM-a | — | 1,43 pp | 1,43 pp |
| Trainable params | 109.485.314 | 394.754 (-99,64%) | 394.754 (-99,64%) |
| Waktu latih | 136,06 s | 21,64 s (-84,1%) | 21,64 s (-84,1%) |
| Latency per sampel | 6,67 ms | 6,67 ms | 8,73 ms |
| Memori GPU inferensi (bobot / puncak) | 429,5 / 436,1 MB | 431,5 / 438,1 MB | 431,5 / 438,1 MB |
| Kriteria sukses | — | 3/3 | 3/3 |

Kandidat #2 pada test: RM-a #1 0,968762, RM-b #30 0,944391, RM-c #1597 0,952780.

**Temuan untuk Bab 4:**

1. **RM-c identik dengan RM-b di test**: 0 dari 1.405 prediksi berbeda. Dengan alpha
   0,1 dan k=1, retrieval hanya bisa membalik prediksi dengan p_bert 0,50-0,56. Konsisten
   dengan validation (gain +0,12 pp, di bawah ambang derau). RAC berperan sebagai
   pengoreksi head lemah, bukan penambah head terkuat.
2. **Urutan kandidat seri tidak stabil di test**: kandidat #2 RM-a dan RM-c lebih baik
   di test (+0,53 dan +0,36 pp). Juara tidak ditukar (test tidak dipakai untuk seleksi);
   selisih di dalam ambang seri tak terbedakan dengan satu seed.
3. **Penurunan val -> test lebih besar pada RM-b/RM-c** (1,93 / 2,04 pp vs 1,51 pp
   RM-a): epoch terbaik RM-b dipilih dari 100 epoch yang berderau (train loss mendekati
   nol, val F1 berfluktuasi sekitar ±0,3 pp per epoch), sehingga val F1-nya optimistis.
4. **RM-b lebih sering melewatkan komentar judi**: recall judi 0,902 vs 0,957, precision
   setara.
5. **Saat inferensi ketiga strategi setara** dalam latency dan memori (encoder
   mengambil 98,8% latency RM-a dan 97,8% RM-b). RM-c menambah retrieval FAISS 1,05 ms dan fusi 0,11 ms;
   encoder RM-c konsisten sekitar 0,8 ms lebih lambat daripada encoder RM-b yang
   identik, dugaan kontensi thread OpenMP FAISS (belum dibuktikan). Efisiensi RM-b/RM-c
   sepenuhnya di sisi training: parameter -99,64%, waktu -84,1%, memori latih -85,3%.
6. **Biaya indeks FAISS**: bangun 3,6 ms, 19,3 MB, 6.587 vektor dari train. Waktu
   penelusuran batch per query (07) tidak menunjukkan tren terhadap k (53-124 us,
   berderau karena CPU instance dipakai bersama); angka retrieval resmi dari 05.

**Catatan proses (untuk transparansi Bab 3/4):**

- 03b sempat dijalankan dengan Run All sebelum CSV tahap lanjut diperbarui (run
  #7-#27 dari grid kampanye lama). Hasil RM-b dipindah ke luar repo dan RM-b diulang
  dari nol; tahap 1 dan 1B identik sampai 6 desimal (RM-b deterministik).
- Kandidat #2 RM-c awal (#2047, uniform pada k=1) identik secara prediksi dengan #1.
  Aturan pengecualian ditambahkan (sejajar varian epoch RM-b) dan kandidat ditetapkan
  ulang sebelum test dibuka; versi lama disimpan di
  `candidates_superseded_k1_weighting.json`.
- 05 dijalankan tiga kali dalam sesi yang sama; F1 test identik di ketiganya. Run
  kedua dan ketiga memperbaiki pengukuran memori inferensi (sebelumnya seluruh model
  berada di GPU sekaligus) dan membekukan RM-a saat inferensi (parameter
  `requires_grad` membuat autocast menyimpan salinan fp16 bobot, +186 MB). Angka
  latency dan memori resmi dari run ketiga (09:14:05).
- 06 dijalankan ulang setelah `fonts-liberation` dipasang supaya gambar memakai
  Liberation Serif (metrik identik dengan Times New Roman).

## Perubahan 2026-09-30

**Refactor ke gaya `indonesian-rag-retrieval-benchmark`.** Setiap fungsi hidup di
notebook yang memakainya; fungsi yang dipakai beberapa notebook disalin (daftar di
`CLAUDE.md`). `src/`, `tests/`, `pyproject.toml`, dan `.env.example` dihapus. Inti
metodologi dipertahankan: grid per tahap, aturan seri 0,15 pp, putusan juara RM-c
dengan bootstrap berpasangan, `candidates.json` berhash, gate hardware di 03a/05, dan
gate checksum di 02. Yang dibuang: resume otomatis, Rumus 1-4, `FigureReporter` per
run, dan `HASIL.xlsx`/`RunMerger`. Setiap notebook diverifikasi ekuivalen dengan kode
lama pada subset kecil (bobot RM-a selisih 0,0; embedding, head, run RM-c, putusan,
kandidat, metrik 05, dan seluruh 21 sheet 06 identik; gambar identik per piksel).

**Decode entitas HTML sebelum kunci dedup (Tahap 2 notebook 02).** `textOriginal`
memuat 4 baris dengan 5 entitas (`&amp;` x3, `&quot;` x2) dan tanpa tag HTML (bagian 7
notebook 01). Entitas itu menyembunyikan duplikat: "Pulau777, penuh warna &amp;
keseruan" (3 baris) dan versi `&` (3 baris) kini tergabung. Kolom `textOriginal`
tetap disimpan mentah; decode hanya dipakai untuk kunci dedup dan `clean_text`.
Split dibangun ulang (angka di bagian Dataset). 02 dijalankan dua kali: run pertama
menulis split, run kedua melaporkan `identik` di gate checksum.

**Statistik token setelah pembersihan** dipindah ke Tahap 6 notebook 02: panjang
token per kelas dan sumber [UNK] (emoji/simbol, kata yang tertelan utuh menjadi satu
[UNK]).

**Grid tahap lanjut adaptif.** Grid tahap 1 RM-a (24 run), tahap 1 RM-b (4 run), dan
grid RM-c (66 `alpha x k` per head) dijalankan apa adanya. CSV tahap lanjut
(`RMA_TUNING_GRID_STAGE2`, `RMB_TUNING_GRID_STAGE1B/2/3`) masih berisi keputusan
kampanye lama dan disusun ulang dari pemenang baru, dengan sumbu dan nilai yang sama.
Panduan penyusunannya ada di markdown sebelum setiap sel tahap lanjutan di 03a, 03b,
dan 03c. Sel tahap 2 di 03a kini membaca ulang CSV saat dijalankan. Selama kampanye
ditambahkan dua tahap perluasan karena pemenang di tepi grid:
`RMB_TUNING_GRID_STAGE2B.csv` (epoch 50/100) dan `RMB_TUNING_GRID_STAGE3B.csv`
(dropout 0,4/0,5); sel keduanya disisipkan manual di 03b instance.

**Perbaikan kode selama kampanye** (seluruhnya sebelum atau tanpa memengaruhi seleksi):
03c mengeluarkan varian `weighting` pada k=1 dari kandidat #2 RM-c; 05 mengukur memori
inferensi per skenario secara terisolasi dan membekukan RM-a saat inferensi; 06
merapikan label Gambar 4.2 dan titik berimpit Gambar 4.8; 07 menambah pemanasan dan
20 pengulangan pada pengukuran penelusuran FAISS.

## Dataset

`data/processed/{train,val,test}.csv` dengan kolom `textOriginal`, `text_clean`,
`label`. **Input model adalah `text_clean`.**

| Tahap | Baris |
|---|---|
| Mentah | 14.237 |
| Setelah buang missing | 14.227 |
| Setelah dedup (kunci NFKC atas teks ter-decode) | 9.411 |
| Dibuang guard anti-kebocoran | 17 |
| Final | 9.394 |

train 6.587 (judi 1.196) / val 1.402 (257) / test 1.405 (256), sekitar 18% kelas
judi, rasio 4,5:1. Class weight dari train `{0: 0.6109, 1: 2.7538}`. Konflik label:
3 grup, 32 baris, kebijakan `assign_1`.

Checksum isi kanonik (12 karakter pertama): train `da68f14363eb`, val `854a544e1ad9`,
test `8352c8d18d91`. Gate Tahap 9 notebook 02 di instance RTX 4090 melaporkan
`identik` untuk ketiga split.

Model dasar `indobenchmark/indobert-base-p2`, `max_length` 128, special token
`[URL]`, `[MENTION]`, `[NUM]`.

## Durasi kampanye RTX 4090

Kalibrasi RM-a (baseline `lr=2e-5, epochs=5, batch=16`) 90 s dengan puncak memori
3,6 GB; grid tahap 1 RM-a (24 run) sekitar 28 menit. Setiap head RM-b 5-38 s. Grid
RM-c 2.046 run selesai dalam 1,7 menit. Seluruh kampanye, dari gate hardware 03a
(07:41) sampai arsip 07 (09:33), berjalan dalam satu boot instance. Instance
pertama ditolak karena driver hanya mendukung CUDA 12.7; instance kedua dipilih
dengan Max CUDA >= 13.0.

## Pelajaran yang tetap berlaku

**`micro_batch` adalah bagian identitas konfigurasi.** Rumus gradien akumulasi
ekuivalen dengan batch besar, tetapi run-nya tidak: `DataLoader` dengan ukuran batch
berbeda mengonsumsi RNG berbeda, sehingga mask dropout dan komposisi batch berubah.
Di kampanye lokal, konfigurasi sama dengan `micro_batch` efektif 8 versus 16 memberi
val F1-macro 0,974873 versus 0,977266. Grid RM-a mengunci `micro_batch = 32` untuk
seluruh sel, dan nilainya tidak boleh berubah di tengah kampanye.

**Pergeseran F1 antar hardware kecil, tetapi pemenang tetap ditentukan ulang.**
Run baseline yang sama di RTX 3090 dan RTX 3050 memberi confusion matrix validation
yang persis sama pada epoch terbaiknya masing-masing, walau kurvanya berbeda.

**Gate split membandingkan isi, bukan byte.** SHA-256 byte berkas berbeda antara
Windows dan Linux karena pandas menulis ujung baris sesuai OS dan git
`core.autocrlf=true` menormalkan CR di dalam `textOriginal` (2026-09-20). Gate kini
menyeragamkan ujung baris di setiap sel ke LF sebelum di-hash.

**Hasil run tidak disimpan di git** (sejak 2026-09-20). Hasil yang ikut ter-clone
tercampur dengan kampanye baru. Amankan hasil lewat sel "Arsipkan hasil" di
`07_archive.ipynb`.

**Retrieval memakai pencarian exact.** `faiss.IndexFlatIP` atas embedding yang
dinormalisasi L2 (cosine), indeks hanya dari train. Pada 6.587 vektor biayanya kecil,
hasilnya deterministik, dan tidak ada hyperparameter ANN yang mengganggu perbandingan.

## Riwayat kampanye lama (tidak dipakai untuk Bab 4)

- **RTX 3090, Vast.ai** (hingga 2026-09-14): angka tersimpan di tag
  `hasil-vast-2026-09-14`. Arsip lokalnya (`outputs/_archive_vastai_2026-09-30/`)
  dihapus setelah kampanye RTX 4090 terverifikasi (2026-09-30).
- **RTX 3050 Laptop, lokal** (2026-09-02 sampai 2026-09-05): RM-a 30 run, RM-b 27 run,
  RM-c 67 run, kriteria sukses RM-b dan RM-c 3/3. Dijalankan di split lama (9.395
  baris) dan struktur kode lama.
- Rumus fusi 1-4 dan eksplorasi RM-c lintas rumus diuji lalu dikeluarkan dari
  kampanye (2026-09-24); RM-c memakai satu rumus fusi linear level probabilitas.

## Langkah berikutnya

1. Tulis Bab 4 dari `outputs/tuning/metrics/` dan `outputs/tuning/artifacts/`
   (Gambar 4.1-4.8, `data_tabel.xlsx`); temuan dan catatan proses di atas menjadi
   bahan pembahasan.
2. 06 bisa dijalankan ulang di mesin lokal tanpa GPU bila gambar perlu diubah; pasang
   font Liberation Serif lebih dulu supaya tampilannya sama dengan versi instance.

## Aturan validitas Bab 4

Angka efisiensi (waktu latih, latency, peak memory) hanya sah bila berasal dari
satu hardware dan satu sesi. F1 tidak bergantung hardware.
