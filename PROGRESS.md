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
| Kampanye tuning RM-a / RM-b / RM-c | Belum, menunggu run di RTX 4090 | `outputs/tuning/` (kosong) |
| Benchmark final (test, satu sesi) | Belum | `outputs/tuning/metrics/` |
| Artefak Bab 4 (gambar, data tabel) | Belum | `artifacts/` dari `06_artifacts.ipynb` |
| Penulisan Bab 4 | Belum | menunggu angka kampanye baru |

## Keadaan sekarang

Kode dan data siap untuk kampanye ulang di satu instance Vast.ai **RTX 4090 24 GB**
(03a sampai 05 di mesin yang sama). Seluruh angka kampanye sebelumnya, baik RTX 3090
(Vast.ai) maupun RTX 3050 (lokal), **tidak berlaku untuk Bab 4**: split berubah
(decode entitas HTML), sehingga riwayat run lama menjadi yatim. Langkah penyiapan
instance ada di README, bagian "Menjalankan di instance Vast.ai (RTX 4090)".

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
dan 03c. Sel tahap 2 di 03a kini membaca ulang CSV saat dijalankan.

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
test `8352c8d18d91`. Gate Tahap 9 notebook 02 di instance harus melaporkan `identik`.

Model dasar `indobenchmark/indobert-base-p2`, `max_length` 128, special token
`[URL]`, `[MENTION]`, `[NUM]`.

## Rencana kampanye RTX 4090

1. Siapkan instance sesuai README (Python 3.13, torch 2.12.1 `cu130`, kernel
   `indobert-rac`).
2. Jalankan 02: gate harus `identik`.
3. 03a: gate hardware mencatat `hardware.json`, run kalibrasi, grid tahap 1. Kirim
   `runs_rma.csv` untuk menyusun CSV tahap 2 (dan tahap perluasan bila pemenang di tepi).
4. 03b: tahap 1, lalu 1B (kondisional), 2, 3; CSV tiap tahap disusun dari pemenang
   tahap sebelumnya.
5. 03c: grid seluruh head, putusan juara, cek `weighting=uniform`, `candidates.json`.
6. 05, 06, 07; unduh `hasil_*.tar.gz` sebelum instance dihancurkan.

**Perkiraan waktu.** Baseline RM-a (`lr=2e-5, epochs=5, batch=16`) makan 143,2 s di
RTX 3090 (kampanye lama). Dengan 4090 sedikit lebih cepat, grid tahap 1 RM-a
(128 epoch) diperkirakan di bawah satu jam, RM-b dan RM-c masing-masing hitungan
menit. Angka pastinya dari run kalibrasi di 03a.

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
  `hasil-vast-2026-09-14`. Notebook ber-output, figur EDA, dan checkpoint final
  diarsip lokal di `outputs/_archive_vastai_2026-09-30/` (tidak ikut git) sampai
  kampanye RTX 4090 terbukti berhasil, lalu dihapus.
- **RTX 3050 Laptop, lokal** (2026-09-02 sampai 2026-09-05): RM-a 30 run, RM-b 27 run,
  RM-c 67 run, kriteria sukses RM-b dan RM-c 3/3. Dijalankan di split lama (9.395
  baris) dan struktur kode lama.
- Rumus fusi 1-4 dan eksplorasi RM-c lintas rumus diuji lalu dikeluarkan dari
  kampanye (2026-09-24); RM-c memakai satu rumus fusi linear level probabilitas.

## Langkah berikutnya

1. Jalankan kampanye di RTX 4090 sesuai rencana di atas.
2. Setelah kampanye berhasil, hapus `outputs/_archive_vastai_2026-09-30/` dan
   `models/*.pt` lama.
3. Tulis Bab 4 dari `outputs/tuning/metrics/` dan `artifacts/`.

## Aturan validitas Bab 4

Angka efisiensi (waktu latih, latency, peak memory) hanya sah bila berasal dari
satu hardware dan satu sesi. F1 tidak bergantung hardware.
