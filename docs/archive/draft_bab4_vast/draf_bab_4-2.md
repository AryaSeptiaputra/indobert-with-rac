# 4.2 Pre-processing

Sebelum data digunakan untuk pelatihan model, dilakukan serangkaian tahap *preprocessing*
terhadap data mentah hasil pengumpulan (`dataset/raw/data_labeling.csv`, 14.237 baris
komentar YouTube berlabel). Subbab ini menyajikan **hasil kuantitatif** dari pipeline
*preprocessing* tersebut — dampaknya terhadap jumlah data, kualitas representasi teks, dan
distribusi kelas final — sebagai dasar empiris sebelum masuk ke hasil pemodelan pada §4.3.

## 4.2.1 Reduksi Jumlah Data per Tahap

| Tahap | Jumlah baris | Perubahan |
|---|---|---|
| Data awal (mentah) | 14.237 | — |
| Setelah pembuangan baris kosong (`textOriginal`/`label`) | 14.227 | −10 |
| Setelah deduplikasi (kunci NFKC-*exact*) | 9.412 | −4.815 |
| Setelah *guard* anti-*leakage* lintas-*split* | **9.395** | −17 |

Tabel 4.4 Reduksi jumlah baris per tahap *preprocessing* (`dataset/processed/metadata.json`)

Deduplikasi merupakan tahap dengan reduksi terbesar (−4.815 baris, ±34% dari data setelah
pembuangan baris kosong), mengindikasikan tingginya duplikasi konten pada komentar promosi
judi daring — konsisten dengan pola *spam* yang menyebar komentar identik atau nyaris identik
secara berulang. Kunci deduplikasi menggunakan bentuk **NFKC-*exact*** (bukan teks mentah apa
adanya), yang mampu menangkap variasi obfuskasi Unicode pada nama-nama brand judi (mis. huruf
*fullwidth*, *double-struck*, atau *enclosed*) sebagai representasi yang setara.

Tahap *guard* anti-*leakage* kedua membuang 17 baris tambahan: setelah teks dibersihkan
(`clean_text`) dan di-*placeholder*-kan, sejumlah kecil baris di *val*/*test* menjadi identik
dengan baris di *train* (mis. templat *spam* yang berbeda hanya pada URL/angka yang sudah
diseragamkan menjadi `[URL]`/`[NUM]`) — baris duplikat ini dibuang khusus dari *val*/*test*
(data *train* dipertahankan) untuk menjaga independensi antar-*split*, krusial terutama untuk
validitas indeks FAISS pada RM-c (§4.3.3).

## 4.2.2 Dampak Normalisasi NFKC

Normalisasi NFKC menjadi fondasi utama pipeline *preprocessing*, dengan dampak terukur pada
tingkat token tak-dikenal (`[UNK]`) dari tokenizer IndoBERT:

| Kondisi | Tingkat `[UNK]` pada baris kelas judi |
|---|---|
| Sebelum normalisasi NFKC | 90,5% |
| Setelah normalisasi NFKC | 53,6% |

Tabel 4.5 Dampak normalisasi NFKC terhadap tingkat token `[UNK]` (`EDA_REPORT_BAGIAN2.md`)

Penurunan tajam ini menunjukkan bahwa mayoritas obfuskasi Unicode pada teks promosi judi
(digunakan untuk menghindari deteksi berbasis kata kunci) berhasil dipulihkan ke bentuk
karakter standar, sehingga representasi subword oleh tokenizer IndoBERT menjadi jauh lebih
bermakna dibanding tanpa normalisasi.

## 4.2.3 Penanganan Konflik Label dan Transformasi Teks

Pada proses deduplikasi, ditemukan 3 kelompok baris (32 baris total) dengan label yang tidak
konsisten dalam satu kelompok duplikat — seluruhnya diselesaikan dengan kebijakan
**`assign_1`** (mayoritas anggota kelompok memang berlabel judi).

Transformasi teks (`clean_text`) yang diterapkan setelah *split* mencakup: normalisasi NFKC,
penggantian URL menjadi `[URL]`, mention (`@\w+`) menjadi `[MENTION]`, dan angka berdiri
sendiri menjadi `[NUM]` — sementara *huruf* pada nama brand alfanumerik (mis. `DORA77`)
dipertahankan utuh karena menjadi sinyal utama klasifikasi. Transformasi yang **tidak**
dilakukan mencakup *lowercasing* manual, penghapusan emoji, penghapusan tanda baca, maupun
pembuangan komentar pendek/non-Indonesia/*outlier* panjang — keputusan ini didasarkan pada
analisis EDA yang menunjukkan elemen-elemen tersebut tidak memberi sinyal gangguan berarti,
sementara membuangnya berisiko menghilangkan informasi yang justru relevan (lihat
`EDA_REPORT_BAGIAN3.md`).

## 4.2.4 Dataset Final

Hasil akhir *preprocessing* — 9.395 baris dengan rasio kelas ~4,5:1 dan pembagian
*train*/*val*/*test* 70:15:15 — dirangkum secara lengkap pada Tabel 4.2 dan Tabel 4.3 di
§4.1.2. Artefak keluaran tersimpan pada `dataset/splits/{train,val,test}.csv` (kolom
`textOriginal`, `text_clean`, `label`) dan `dataset/processed/metadata.json`.
