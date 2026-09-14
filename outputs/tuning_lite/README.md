# outputs/tuning_lite/ — eksplorasi varian encoder IndoBERT-lite

**Status: arsip kode dan hasil. TIDAK dipakai sebagai sumber angka dokumen
penelitian.** Angka Bab 4 tetap berasal dari `outputs/tuning/`
(`indobenchmark/indobert-base-p2`). Folder ini disimpan supaya eksplorasinya
bisa ditelusuri ulang, bukan untuk dikutip.

## Pertanyaan yang diuji

Apakah kontribusi RAC (selisih F1 RM-c terhadap RM-b) berubah besar bila encoder
beku diganti `indobenchmark/indobert-lite-base-p2` (ALBERT, 11,7 jt parameter
unik) menggantikan `indobert-base-p2` (110 jt)? Ketiga skenario RM-a/RM-b/RM-c
dijalankan ulang penuh memakai grid yang sama persis di `tuning_grids/`.

## Isi

| File | Isi |
|------|-----|
| `runs_{rma,rmb,rmc}.csv` | 28 / 27 / 67 run. Kolom `model_name` seluruhnya `indobenchmark/indobert-lite-base-p2`. |
| `best.json` | Juara per skenario menurut F1-macro validation. |
| `metrics/final_comparison.csv` | Evaluasi TEST ketiga skenario, satu sesi. |
| `metrics/success_criteria.csv` | RM-b dan RM-c sama-sama 3/3 kriteria. |
| `faiss_index/` | Biaya bangun & telusur indeks FAISS untuk embedding lite. |
| `history/`, `figures/`, `tuning_summary.json`, `hardware.json` | Sama seperti konvensi `outputs/tuning/`. |

Workbook `HASIL_lite.xlsx` di root repo bersifat turunan dan gitignored, sama
seperti `HASIL.xlsx`.

## Hasil ringkas

| | base | lite |
|---|---|---|
| RM-a test F1-macro | 0,9561 | 0,9561 |
| RM-b test F1-macro | 0,9525 | 0,9394 |
| RM-c test F1-macro | 0,9525 | 0,9446 |
| Kontribusi RAC di test (RM-c − RM-b) | 0,00 pp | +0,53 pp |
| RM-a trainable params | 109.485.314 | 11.685.378 |
| RM-a waktu latih | 1.560 s | 460 s |
| RM-a peak memory inferensi | 1.049 MB | 119 MB |

Jawaban atas pertanyaan di atas: **tidak ada perbedaan besar.** Kontribusi RAC
tetap kecil pada kedua encoder (di validation +0,13 pp base versus +0,23 pp
lite). Biaya retrieval RAC juga terbukti tidak bergantung ukuran encoder —
indeks kedua kampanye identik (6.588 vektor x 768 dimensi, 19,301 MB) karena
dimensi embedding ALBERT-lite-base sama dengan BERT-base.

## Mengapa tidak dipakai untuk Bab 4

1. **Juara RM-a lite ada di tepi grid.** Rentang `lr` yang diuji [1e-5 ... 5e-5]
   dan pemenangnya `lr=1e-5`, yaitu nilai terendah. Aturan seleksi di
   `CLAUDE.md` menyebut pemenang di tepi grid adalah sinyal untuk melebarkan
   rentang, bukan untuk dikunci. Kampanye base tidak punya masalah ini
   (pemenangnya `lr=3e-5`, di tengah rentang).
2. **Degradasi validation ke test dua kali lipat.** RM-b lite turun 2,95 pp
   (0,9688 ke 0,9394) versus RM-b base 1,47 pp; RM-c lite turun 2,65 pp versus
   RM-c base 1,60 pp. Di validation lite tampak unggul, di test justru kalah —
   gejala seleksi hyperparameter yang mengejar validation.
3. **Argumen inti skripsi melemah.** Gap F1 terhadap RM-a: base 0,37 pp untuk
   RM-b maupun RM-c, lite 1,68 pp dan 1,15 pp.
4. **Grid-nya dipinjam mentah dari desain base**, belum dikalibrasi ulang untuk
   dinamika cross-layer parameter sharing ALBERT, dan tidak punya dokumen
   rasional setara `tuning_grids/*.md`.

## Cara menjalankan ulang

Notebook `04_tuning_campaign_lite.ipynb`, `05_final_benchmark_lite.ipynb`, dan
`06_analysis_export_lite.ipynb` adalah salinan notebook base dengan sel
konfigurasi yang mengarah ke `out_dir=outputs/tuning_lite` dan
`model_name="indobenchmark/indobert-lite-base-p2"`, plus tujuan tulis yang
dinamespace supaya artefak base tidak tertimpa (`faiss_index` ke dalam
`out_dir`, workbook ke `HASIL_lite.xlsx`, checkpoint ke `models/lite/`).

Kampanye ini sendiri dieksekusi headless lewat skrip setara isi notebook, bukan
dari dalam Jupyter, sehingga sel-sel notebook tersimpan tanpa output.

`load_tokenizer` (`src/models/comment_dataset.py`) wajib untuk repo lite:
`AutoTokenizer` salah mengenali repo ini sebagai ALBERT/SentencePiece sehingga
vocab runtuh ke 5 token, lalu kode jatuh balik ke `BertTokenizer` WordPiece
(30.002 token). Jalur fallback itu sebelumnya belum pernah diuji terhadap repo
lite yang sungguhan; kampanye ini yang pertama memverifikasinya.
