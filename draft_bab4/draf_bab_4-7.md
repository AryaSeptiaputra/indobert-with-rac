# 4.7 Analisis Trade-off Performa vs Efisiensi

Subbab ini mensintesiskan temuan §4.4 (performa test), §4.5 (efisiensi komputasi), dan §4.6
(kriteria sukses) menjadi argumen utama penelitian ini: **seberapa besar biaya performa yang
harus dibayar untuk memperoleh efisiensi komputasi, dan apakah trade-off tersebut sepadan?**

## 4.7.1 Biaya Performa vs Penghematan Sumber Daya

| | RM-b vs RM-a | RM-c vs RM-a |
|---|---|---|
| Biaya performa (gap F1-macro test) | −1,21pp | −1,10pp |
| Penghematan *trainable parameters* | +99,28% | +100% |
| Penghematan waktu latih | +85,64% | +100% |

Tabel 4.13 Ringkasan trade-off performa vs efisiensi

Ditinjau dari sisi *pelatihan*, trade-off ini sangat berpihak pada strategi ringan: biaya
performa yang dibayar (≤1,21 poin persentase F1-macro) jauh lebih kecil dibanding proporsi
sumber daya yang dihemat (>99% parameter, >85% waktu latih). Dengan kata lain, RM-b dan RM-c
memperoleh **>98% dari performa RM-a dengan <1% dari biaya pelatihannya** — argumen yang kuat
untuk skenario di mana model perlu sering dilatih ulang (mis. adaptasi berkala terhadap pola
promosi judi yang terus berubah) atau dilatih pada perangkat keras terbatas.

## 4.7.2 RM-c vs RM-b: Kontribusi RAC yang Tipis tapi Konsisten

RM-c unggul tipis dari RM-b murni (+0,11pp F1-macro test, §4.4.1) tanpa parameter maupun
waktu latih tambahan sama sekali — namun selisih ini berada **di bawah ambang seri (0,15pp)**
yang digunakan sepanjang proses tuning penelitian ini (§4.3). Kontribusi RAC pada kondisi
head performa-terbaik karena itu dibaca sebagai **bonus tanpa risiko**, bukan pendorong
utama argumen kompetitif RM-c — signifikansi statistiknya belum diuji secara formal karena
keterbatasan satu *seed*/*split* pengujian.

Nuansa ini berubah signifikan pada kondisi head yang jauh lebih murah: analisis ablasi pada
§4.3.3 menunjukkan kontribusi RAC melonjak menjadi +4,51pp (val) ketika dipasangkan dengan
head *linear* (1.538 parameter, 513× lebih murah dari head MLP/1024), dengan performa test
yang bahkan sedikit melampaui RM-c konfigurasi resmi. Temuan ini memperluas argumen
efisiensi penelitian ini: RAC tidak hanya menjadi pelengkap tipis bagi head yang sudah kuat,
tetapi berpotensi menjadi **substitusi kapasitas** bagi head yang jauh lebih ringan.

## 4.7.3 Batasan: Latency Inferensi Belum Ikut Membaik

Argumen efisiensi di atas perlu diimbangi dengan temuan §4.5.3: **latency inferensi RM-b/RM-c
tidak ikut membaik** dibanding RM-a, karena *forward pass* melalui encoder BERT-base 110 juta
parameter tetap dijalankan penuh pada ketiga skenario saat prediksi. Implikasinya, klaim
efisiensi pada penelitian ini **berlaku kuat untuk konteks pelatihan/adaptasi model**, namun
belum menjawab kebutuhan efisiensi pada konteks *serving* volume tinggi yang sensitif
terhadap *latency* per prediksi — sebuah batasan yang perlu dinyatakan secara eksplisit agar
klaim trade-off tidak ditafsirkan melampaui bukti yang ada.

## 4.7.4 Simpulan Trade-off

Secara keseluruhan, RM-b dan RM-c terbukti menjadi pilihan yang **praktis dan sepadan** untuk
skenario adaptasi model dengan sumber daya pelatihan terbatas, dengan RM-c memberi keunggulan
tipis tambahan tanpa biaya ekstra dibanding RM-b. Namun demikian, untuk kasus penggunaan yang
menuntut efisiensi *inferensi* (bukan pelatihan), ketiga strategi pada penelitian ini
memiliki karakteristik biaya prediksi yang serupa — sebuah arah eksplorasi lanjutan yang
relevan untuk penelitian berikutnya.
