# 4.1 Lingkungan dan Konfigurasi Eksperimen

Seluruh eksperimen tuning dan pengujian akhir pada penelitian ini dijalankan pada satu unit
GPU sewaan (Vast.ai) dalam **satu sesi tunggal**, khususnya untuk seluruh pengukuran yang
berkaitan dengan efisiensi komputasi (waktu latih, *peak memory*, dan *latency* inferensi).
Konsistensi lingkungan eksekusi ini merupakan prasyarat mutlak agar perbandingan efisiensi
antar-strategi (RM-a, RM-b, RM-c) tetap valid — mencampur hasil dari perangkat keras atau
sesi yang berbeda dapat menimbulkan bias pengukuran yang tidak berkaitan dengan karakteristik
strategi adaptasi itu sendiri, melainkan semata perbedaan beban kerja perangkat pada saat
pengukuran dilakukan. Metrik performa klasifikasi (F1-score), sebaliknya, bersifat
*hardware-independent* sehingga tetap dapat dibandingkan lintas sesi bila diperlukan.

## 4.1.1 Spesifikasi Perangkat Keras dan Perangkat Lunak

| Komponen | Spesifikasi |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| CUDA | 13.0 |
| PyTorch | 2.12.0+cu130 |
| Transformers (HuggingFace) | 5.14.1 |
| Penyedia | Vast.ai (sewa daring, *on-demand*) |

Tabel 4.1 Spesifikasi lingkungan eksperimen (`results/vast/hardware.json`)

## 4.1.2 Ringkasan Dataset dan Konfigurasi Tokenisasi

Dataset final hasil *preprocessing* (rincian lengkap pada §4.2) terdiri atas 9.395 baris
komentar YouTube berbahasa Indonesia, dibagi menjadi tiga subset dengan skema *stratified
split* 70:15:15 agar proporsi kelas terjaga di setiap subset.

| Split | Jumlah baris | Kelas 0 (non-judi) | Kelas 1 (judi) | Persentase kelas 1 |
|---|---|---|---|---|
| Train | 6.588 | 5.391 | 1.197 | 18,17% |
| Validation | 1.402 | 1.145 | 257 | 18,33% |
| Test | 1.405 | 1.149 | 256 | 18,22% |
| **Total** | **9.395** | **7.685** | **1.710** | **18,20%** |

Tabel 4.2 Distribusi kelas per split (`dataset/processed/metadata.json`)

Rasio ketidakseimbangan kelas sekitar 4,5:1 (kelas non-judi dominan) menjadikan metrik
**F1-macro** sebagai metrik utama evaluasi pada penelitian ini, bukan *accuracy* semata, agar
performa terhadap kelas minoritas (judi) turut terukur secara proporsional. Untuk mengatasi
ketidakseimbangan ini pada level fungsi *loss*, digunakan pembobotan kelas (*class weights*)
yang dihitung dari distribusi *train set*:

| Kelas | Class weight |
|---|---|
| 0 (non-judi) | 0,6110 |
| 1 (judi) | 2,7519 |

Tabel 4.3 Class weights (`dataset/processed/metadata.json`)

Seluruh model menggunakan basis pra-latih **`indobenchmark/indobert-base-p2`** dengan panjang
token maksimum 128 (mencakup ≥99,9% panjang komentar setelah *preprocessing*, lihat §4.2).
Tiga *placeholder* hasil *preprocessing* — `[URL]`, `[MENTION]`, dan `[NUM]` — didaftarkan
sebagai *additional special tokens* pada tokenizer, sehingga setiap model wajib memanggil
`resize_token_embeddings()` sebelum pelatihan maupun inferensi agar ukuran *vocabulary*
konsisten dengan tokenizer yang telah diperluas.
