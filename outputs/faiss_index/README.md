# results/faiss_index/ — pengukuran modul indeks FAISS (RM-c)

Dihasilkan oleh `python scripts/benchmark_faiss_index.py` (menjalankan fungsi asli
`src/rac.py::build_faiss_index` & `retrieve`, bukan reimplementasi).

| File | Isi |
|------|-----|
| `faiss_index_build.csv` | 1 baris per sesi pengukuran: waktu normalisasi L2 / `index.add` / total (mean, std, min, max atas `repeats`), ukuran indeks di memori & di disk, delta RSS, waktu `write_index`/`read_index`, komposisi label, plus identitas host. |
| `faiss_search_latency.csv` | 1 baris per (split query × k): total ms, ms/query, QPS, rata-rata similaritas top-1 & top-k. |
| `train_index.faiss` | Indeks hasil serialisasi (gitignored lewat pola `*.faiss`). |

Catatan penting:

- **Indeks dibangun HANYA dari embedding train** (anti-leakage, sesuai desain RAC di
  `src/rac.py`); val/test hanya berperan sebagai query.
- `IndexFlatIP` = pencarian **eksak** (bukan approximate) dengan inner product atas embedding
  yang sudah di-L2-normalisasi → setara cosine. Tidak ada tahap training indeks
  (`is_trained=True` sejak awal), sehingga "pembangunan indeks" = normalisasi + `add`.
- **Aturan validitas Bab 4 tetap berlaku**: angka waktu/memori hanya sebanding dalam satu
  host + satu sesi. Kolom `host`, `cpu`, `platform`, `faiss_threads`, `faiss_version` dicatat
  di tiap baris — jangan campur baris lintas-host dalam satu tabel efisiensi. Angka pada
  baris pertama diukur di laptop (CPU), **bukan** sesi RTX 3090 yang menjadi sumber
  `results/vast/metrics/inference_benchmark.csv`.
- Script hanya **menambah** baris (append) dan tidak pernah menyentuh `results/vast/`.
