# 4.8 Ringkasan Bab

Bab ini menyajikan rantai bukti dari kualitas data hingga verdict akhir *trade-off*
performa-vs-efisiensi tiga strategi adaptasi IndoBERT yang dibandingkan pada penelitian ini:

1. **Kualitas data (§4.2)** — pipeline *preprocessing* berhasil mereduksi data mentah bising
   (14.237 baris) menjadi dataset bersih bebas-*leakage* (9.395 baris), dengan dampak
   terukur pada penurunan tingkat token `[UNK]` (90,5% → 53,6%) berkat normalisasi NFKC.
2. **Eksplorasi hyperparameter (§4.3)** — RM-a mengonfirmasi hyperparameter kanonik IndoNLU
   sebagai konfigurasi optimal (F1-macro test 0,9607); RM-b menunjukkan pilihan arsitektur
   head jauh lebih menentukan daripada tuning hyperparameter (MLP mengungguli linear
   +5,28pp val dengan hanya 0,72% parameter RM-a); RM-c menunjukkan kontribusi RAC
   berbanding terbalik dengan kapasitas head, menutup gap hingga +4,51pp pada head paling
   ringan.
3. **Hasil test set (§4.4)** — peringkat val (RM-a > RM-c > RM-b) konsisten pada test set,
   mengonfirmasi keputusan tuning tidak *overfit* terhadap *validation set*.
4. **Efisiensi komputasi (§4.5)** — RM-b/RM-c hemat >99% parameter dan >85% waktu latih,
   namun *latency* inferensi nyaris tidak membaik karena *forward pass* encoder tetap
   dijalankan penuh saat prediksi.
5. **Kriteria sukses (§4.6)** — RM-b dan RM-c sama-sama memenuhi 3 dari 3 kriteria yang
   ditetapkan (syarat minimal 2 dari 3), memberi verdict kuantitatif bahwa keduanya
   kompetitif terhadap *full fine-tuning*.
6. **Sintesis trade-off (§4.7)** — biaya performa marjinal (≤1,21pp) jauh lebih kecil
   dibanding penghematan sumber daya pelatihan, menjadikan RM-b/RM-c pilihan praktis untuk
   skenario pelatihan/adaptasi terbatas sumber daya — dengan catatan penting bahwa efisiensi
   ini belum menjangkau sisi *latency* inferensi.

Temuan-temuan ini menjadi dasar bagi kesimpulan dan saran penelitian lanjutan pada Bab 5,
khususnya terkait arah eksplorasi untuk mengatasi keterbatasan *latency* inferensi yang
diidentifikasi pada §4.5.3 dan §4.7.3.
