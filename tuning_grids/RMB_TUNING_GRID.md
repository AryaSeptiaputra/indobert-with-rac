# RM-b — Rancangan Eksplorasi Hyperparameter (27 run: 4 Tahap 1 + 2 perluasan + 15 Tahap 2 + 6 Tahap 3)

Dokumen kerja untuk tuning RM-b (frozen encoder + head) lewat `03b_rmb_frozen.ipynb`.
Isi kolom **Alasan** langsung dapat disalin ke field `catatan` di UI. Pola dokumen ini
sengaja disamakan dengan `RMA_TUNING_GRID.md` supaya rigor-nya setara.

> **CSV yang tersedia** (semua siap diunggah lewat mode **Batch → Tempel/unggah tabel
> CSV** di `03b_rmb_frozen.ipynb`, kolom: `head_arch,hidden_dim,epochs,lr,dropout,
> weight_decay,batch,catatan`):
> - `RMB_TUNING_GRID.csv` — 4 baris Tahap 1 (tak bergantung hasil apa pun).
> - `RMB_TUNING_GRID_STAGE1B.csv` — 2 baris perluasan Tahap 1 (`hidden_dim` 768/1024),
>   dibuat setelah run pertama menunjukkan 512 menang di tepi rentang.
> - `RMB_TUNING_GRID_STAGE2.csv` — 15 baris Tahap 2, dibuat setelah pemenang Tahap 1
>   diketahui (`head_arch=mlp, hidden_dim=1024`).
> - `RMB_TUNING_GRID_STAGE3.csv` — 6 baris Tahap 3, dibuat setelah pemenang Tahap 2
>   diketahui (`lr=1e-3, epochs=10`).

---

## Metode: 3 tahap mengikuti struktur ketergantungan field

RM-b jauh lebih murah dari RM-a — melatih head di atas fitur beku (`train_eval_rmb`,
`src/services/training.py:179-217`), hitungan **detik**/run, bukan menit. Tapi murahnya komputasi
**tidak** menghapus risiko *overfitting ke validation set* (~1.402 sampel) — makin banyak
konfigurasi dibandingkan, makin besar peluang pemenang menang karena kebetulan, persis
alasan `RMA_TUNING_GRID.md` menolak grid 5-sumbu penuh untuk RM-a. Jadi kampanye ini
**tetap dijaga di skala ~26 run**, bukan dibesarkan cuma karena komputasinya murah.

Partisi tahap ikuti struktur ketergantungan nyata di kode, bukan asumsi:

- **`head_arch` × `hidden_dim` terkopel struktural** (`build_head`, `src/models/heads.py:157-165`):
  `hidden_dim` cuma dipakai `MLPHead`; `FrozenHead` (linear) tak punya parameter itu sama
  sekali. Tak masuk akal digrid bersama field lain sebelum arsitektur diputuskan.
- **`lr` × `epochs` terkopel kuat dengan arsitektur** — Peters, Ruder, & Smith (2019)
  *"To tune or not to tune? Adapting pretrained representations to diverse tasks"*
  (dikutip di `src/services/training.py:39`): pada rezim *feature-based* (encoder beku), head yang
  diinisialisasi acak butuh **lr lebih tinggi** dan **epoch lebih banyak** dibanding
  fine-tuning penuh — sebab itu `RMB_DEFAULT` lr-nya 2e-4 (10× lipat RM-a) dan rentang
  epoch di sini diuji jauh lebih lebar (5–30) dari RM-a (3–8).
- **`dropout`, `weight_decay`, `batch` relatif independen** — regularisasi/gradient-noise
  di atas head yang jauh lebih kecil dari RM-a (linear: `(768+1)×2` ≈ 1.538 parameter;
  MLP hidden_dim=256: ≈197 ribu parameter — keduanya jauh di bawah 109,5 juta RM-a) →
  coordinate descent di sel pemenang, sama semangatnya dengan `warmup_ratio`/`weight_decay`
  di Tahap 2 RM-a.

**Ukuran train:** N = 6.588 → `ceil(6588/32) = 206` step/epoch (batch dikunci 32 di
Tahap 1–2; Tahap 3 menguji batch 16/64 sebagai salah satu knop independen).

---

## Tahap 1 — Keputusan arsitektur head (4 run, TANPA ketergantungan hasil)

**Dikunci untuk semua sel:** `epochs=5, lr=2e-4, dropout=0.1, weight_decay=0.01, batch=32`
(= `RMB_DEFAULT` penuh, `src/services/training.py:40-41`) — hanya `head_arch`/`hidden_dim` divariasi.

Motivasi: sesi tuning lama (sebelum clean-slate, hasilnya sudah dihapus — dicatat di
`PROGRESS.md`) menemukan "ganti head linear → MLP = lompatan terbesar (val 0,938→0,956)".
Karena ini re-tuning dari nol, temuan lama **tidak boleh diasumsikan berulang begitu
saja** — MLP tetap wajib diuji ulang, dengan `linear` sebagai kontrol wajib. Houlsby dkk.
(2019) *"Parameter-efficient transfer learning for NLP"* jadi rujukan tambahan: kapasitas
kecil tambahan (mirip *adapter*) di atas representasi beku bisa memberi lompatan performa
berarti tanpa mengorbankan efisiensi parameter — jadi masuk akal MLP unggul.

| # | head_arch | hidden_dim | Alasan kombinasi ini diuji |
|---|---|---|---|
| 1 ★ | linear | — | Baseline kanonik RM-b (`RMB_DEFAULT`) — kontrol wajib re-tuning dari nol, titik acuan seluruh grid |
| 2 | mlp | 128 | Kapasitas kecil — cukupkah sedikit non-linearitas tanpa menambah param berarti? |
| 3 | mlp | 256 | Kapasitas default (`RMB_DEFAULT["hidden_dim"]`) — juga yang dipakai RM-c nanti bila MLP menang |
| 4 | mlp | 512 | Kapasitas besar — mendekati batas wajar sebelum jumlah param tumbuh signifikan |

★ = baseline kanonik `RMB_DEFAULT`, **dijalankan pertama** sebagai sanity check.

**Cara jalan:** unggah `RMB_TUNING_GRID.csv` lewat mode Batch → CSV. Pemenang tahap ini
(`head_arch*`, dan `hidden_dim*` bila MLP menang) dikunci untuk Tahap 2.

> **Update pasca-run:** hasil 4 run pertama menunjukkan MLP menang telak dari linear, dan
> `hidden_dim=512` menang **tepat di tepi rentang yang diuji** (128/256/512) dengan
> kenaikan 256→512 (+0,87pp) yang justru lebih besar dari 128→256 (+0,32pp) — belum ada
> tanda melandai. Sesuai aturan seleksi #4 di bawah, ditambah 2 run perluasan
> (`RMB_TUNING_GRID_STAGE1B.csv`: `hidden_dim ∈ {768, 1024}`, field lain sama) sebelum
> `hidden_dim*` dikunci untuk Tahap 2.

---

## Tahap 2 — Grid `lr × epochs` pada arsitektur pemenang (15 run, BERGANTUNG Tahap 1)

> **Update pasca-Tahap 1 (+ perluasan):** pemenang terkunci adalah **`head_arch*=mlp,
> hidden_dim*=1024`** — bukan 512, setelah 2 run perluasan (`RMB_TUNING_GRID_STAGE1B.csv`)
> menunjukkan tren masih naik sampai 1024 (val F1-macro 0,9543), meski sinyal overfit
> pertama juga muncul di situ (`overfit_signal=True`, `best_epoch=4` dari 5) — persis
> yang akan diselidiki lebih sistematis lewat grid epoch di bawah.

**Dikunci:** `head_arch=mlp, hidden_dim=1024`, `dropout=0.1, weight_decay=0.01, batch=32`.
Rentang jauh lebih tinggi & lebar dari RM-a sesuai Peters+2019 di atas:

- `lr ∈ {1e-4, 2e-4, 5e-4, 1e-3}` — 2e-4 = default RMB di tengah; RM-a memakai 1e-5..5e-5,
  di sini 10–20× lebih tinggi karena head diinisialisasi acak (representasi BERT-nya
  sendiri sudah beku, tak perlu lr sekonservatif fine-tuning penuh).
- `epochs ∈ {5, 10, 20, 30}` — RM-a maksimum 8 epoch; di sini diuji sampai 30 karena
  1 epoch RM-b cuma hitungan detik. Checkpoint val-terbaik (`best_epoch`) sudah otomatis
  melindungi dari overfit walau anggaran epoch besar, sama seperti RM-a.

| # | lr | epochs | update (ceil(N/32)×epochs) | Alasan kombinasi ini diuji |
|---|-----|--------|--------|-----------------------------|
| — | 2e-4 | 5 | 1.030 | **Sudah tercakup Tahap 1 run #10** (`hidden_dim=1024, lr=2e-4, epochs=5` → val F1-macro 0,954323) — TIDAK diulang, hemat 1 run redundan |
| 11 | 1e-4 | 5 | 1.030 | lr konservatif (setengah default), anggaran standar — cek underfit |
| 12 | 1e-4 | 10 | 2.060 | lr konservatif + anggaran 2× — beri waktu ekstra untuk konvergen |
| 13 | 1e-4 | 20 | 4.120 | lr konservatif + anggaran besar — batas atas kompensasi lr kecil |
| 14 | 1e-4 | 30 | 6.180 | lr konservatif + anggaran maksimum — masih naik atau sudah plateau di 20? |
| 15 | 2e-4 | 10 | 2.060 | lr default + anggaran 2× — apakah val F1 masih naik setelah epoch 5 (run #10)? |
| 16 | 2e-4 | 20 | 4.120 | lr default + anggaran besar — head 1024 sudah tunjukkan overfit di epoch 5; makin parah di sini? |
| 17 | 2e-4 | 30 | 6.180 | lr default + anggaran maksimum — batas atas eksplorasi epoch pada lr kanonik |
| 18 | 5e-4 | 5 | 1.030 | **lr agresif (2,5×), anggaran standar — konvergensi lebih cepat tanpa rugi F1?** |
| 19 | 5e-4 | 10 | 2.060 | lr agresif + anggaran 2× — rawan overshoot/osilasi loss, cek kestabilan |
| 20 | 5e-4 | 20 | 4.120 | lr agresif + anggaran besar — risiko overfit head 1024 paling tinggi di grid |
| 21 | 5e-4 | 30 | 6.180 | lr agresif + anggaran maksimum — sudut grid paling agresif pada lr sedang |
| 22 | 1e-3 | 5 | 1.030 | lr sangat tinggi (5×), anggaran minim — skenario "cepat & panas", hemat waktu bila berhasil |
| 23 | 1e-3 | 10 | 2.060 | lr sangat tinggi + anggaran 2× — cek stabilitas training (loss meledak/NaN?) |
| 24 | 1e-3 | 20 | 4.120 | **lr sangat tinggi + anggaran besar — kombinasi paling berisiko overfit/divergen** |
| 25 | 1e-3 | 30 | 6.180 | lr sangat tinggi + anggaran maksimum — sudut grid paling ekstrem, batas atas eksplorasi |

**Baca sebagai permukaan:** susun pivot `lr × epochs` (satu tabel, karena `hidden_dim`
sudah dikunci 1024 tahap ini) — apakah lr optimal bergeser saat anggaran epoch bertambah,
atau stabil di satu nilai sepanjang rentang epoch? `reporting.AXES["rmb"] =
("lr", "epochs", "hidden_dim")` sudah otomatis membuat heatmap ini begitu `runs_rmb.csv`
punya ≥2 nilai unik di `lr` dan `epochs` — tak perlu langkah manual. Perhatikan khusus
kolom `overfit_signal` di grid ini — head 1024 sudah terbukti rentan overfit di epoch 5;
kemungkinan besar makin banyak run di sini akan menunjukkan `overfit_signal=True`, yang
justru jadi bahan utama keputusan Tahap 3 (dropout/weight_decay lebih kuat).

**Cara jalan:** `RMB_TUNING_GRID_STAGE2.csv` (folder ini) sudah berisi 15 baris di atas,
siap diunggah lewat mode **Batch → Tempel/unggah tabel CSV**.

---

## Tahap 3 — Coordinate descent `dropout` / `weight_decay` / `batch` (6 run, BERGANTUNG Tahap 2)

> **Update pasca-Tahap 2:** pemenang terkunci adalah **`lr*=1e-3, epochs*=10`**
> (val F1-macro 0,9650; `best_epoch=6` — konvergen jauh sebelum anggaran 10 epoch habis).
> Permukaan `lr×epochs` bersih dan monoton (tak ada kombinasi anomali); lr tinggi di RM-b
> **tidak** memicu instabilitas seperti risiko *catastrophic forgetting* di RM-a, karena
> head-nya baru diinisialisasi acak (tak ada bobot pra-latih yang bisa "dirusak"). Sel
> `lr=1e-3` di `epochs∈{10,20,30}` menghasilkan angka **identik persis** (checkpoint
> selalu berhenti di epoch 6) — bukan overfit yang merugikan, cuma anggaran terbuang.

Di sel pemenang Tahap 2 (`head_arch=mlp, hidden_dim=1024, lr=1e-3, epochs=10`), uji tiap
knop **independen** (2 varian non-default per knop, field lain tetap di nilai pemenang):

| # | Knop diuji | Nilai | Alasan |
|---|---|---|---|
| 26 | dropout | **0,0** (vs default 0,1) | Devlin dkk. (2019) pakai dropout 0,1 sbg standar BERT — uji apakah head sekecil ini justru dirugikan oleh regularisasi tsb |
| 27 | dropout | **0,3** (vs default 0,1) | Regularisasi lebih kuat — dicek meski sel pemenang plateau, bukan menurun, jadi ekspektasi realistis: perbaikan tipis atau tak ada |
| 28 | weight_decay | **0,0** (vs default 0,01) | AdamW *decoupled weight decay* (Loshchilov & Hutter, dikutip `src/services/training.py:36`) — cek apakah regularisasi ini bahkan berpengaruh pada head sekecil ini |
| 29 | weight_decay | **0,1** (vs default 0,01) | Regularisasi 10× lebih kuat — sama semangatnya dengan Tahap 2 RM-a |
| 30 | batch | **16** (vs default 32) | RM-a membuktikan batch=32 unggul dari 16 (lebih cepat & F1 setara/lebih baik) — dicek ulang di RM-b karena training di atas fitur beku (tanpa grad-accum) punya dinamika update berbeda |
| 31 | batch | **64** (vs default 32) | Batch lebih besar lagi — apakah tren "batch besar menang" dari RM-a berlanjut, atau ada titik baliknya? |

**Cara jalan:** `RMB_TUNING_GRID_STAGE3.csv` (folder ini) sudah berisi 6 baris di atas,
siap diunggah lewat mode **Batch → Tempel/unggah tabel CSV**.

---

## Template `catatan` untuk UI

```
Tahap [1/2/3] RM-b, [head_arch=X / lr=Y / epochs=Z / dropout=W / weight_decay=V / batch=U
yang diubah dari baseline]. Tujuan: [peran baris, sesuai kolom Alasan].
```

Isi dari kolom **Alasan** di tabel masing-masing tahap. Contoh:
- **Tahap 1 #3** → `Kapasitas default — juga yang dipakai RM-c nanti bila MLP menang`
- **Tahap 2 #19** → `lr sangat tinggi + anggaran besar — kombinasi paling berisiko overfit/divergen`
- **Tahap 3 #21** → `Uji apakah head sekecil ini justru dirugikan oleh dropout standar BERT`

---

## Aturan seleksi (identik RM-a — sudah terimplementasi di kode, tak perlu langkah manual)

1. **Metrik utama** `val_f1_macro`; **tie-break** `val_f1_judi` (F1 kelas-1/judi).
2. **Ambang seri ≤0,15pp** (`TIE_THRESHOLD_PP`, `src/services/training.py:46`, kolom
   `is_tie_with_best` otomatis terisi di `runs_rmb.csv`) → pilih konfigurasi lebih murah
   (epoch lebih kecil / arsitektur lebih ringan).
3. **Baca grid sebagai permukaan**: pivot `lr × epochs` di Tahap 2 (per `hidden_dim` kalau
   MLP menang Tahap 1) — sudah otomatis lewat `reporting.grid_pivot_and_heatmap`.
4. **Waspadai pemenang di tepi grid** (mis. lr=1e-3 atau epochs=30) → sinyal optimum
   mungkin di luar rentang; pertimbangkan 1–2 run perluasan sebelum mengunci.
5. **Perhatikan `overfit_signal`** (kolom otomatis: `best_epoch < epochs`) — head kecil
   dengan anggaran epoch besar (20–30) rawan overfit; kalau sinyal ini sering `True`,
   jangan buru-buru mengunci epoch besar hanya karena val_f1_macro tertinggi ada di sana.
6. **TEST tidak disentuh** sampai tab Final. Biarkan "Evaluasi TEST juga" tidak dicentang.

---

## Estimasi biaya

RM-b tanpa fine-tuning IndoBERT — cuma head kecil di atas fitur beku yang sudah
diekstraksi sekali (~30 dtk, dipakai ulang seluruh kampanye ini maupun RM-c nanti).
26 run × hitungan detik/run ≈ **beberapa menit total**, jauh di bawah RM-a (49,9 menit
untuk 26 run). Anggaran bukan kendala di sini — batasan sebenarnya tetap risiko
overfitting-ke-val yang dijelaskan di bagian Metode, bukan biaya GPU.

Jalankan **1× Mode SMOKE** sebelum Tahap 1 penuh untuk memastikan pipeline menulis baris
ke `outputs/tuning/runs_rmb.csv` tanpa error (dan ekstraksi fitur beku sukses).

---

## Landasan literatur

| Sumber | Kontribusi ke rancangan |
|---|---|
| Peters, Ruder, & Smith (2019) | *Feature-based transfer*: head acak butuh lr lebih tinggi & epoch lebih banyak dari fine-tuning → dasar rentang lr/epochs Tahap 2 |
| Houlsby dkk. (2019) | *Parameter-efficient transfer learning*: kapasitas kecil tambahan (adapter-like) bisa memberi lompatan performa berarti → motivasi menguji MLP serius di Tahap 1 |
| Devlin dkk. (2019) | Dropout 0,1 sbg standar BERT → titik acuan Tahap 3 (diuji apakah tetap optimal untuk head sekecil ini) |
| Loshchilov & Hutter | AdamW: weight decay *decoupled by design* → dasar `weight_decay` masuk coordinate descent (Tahap 3), sama seperti RM-a |

Temuan internal proyek (bukan literatur akademik, tapi motivasi historis Tahap 1): sesi
tuning RM-b sebelum clean-slate mencatat lompatan val F1 0,938→0,956 saat berpindah dari
head linear ke MLP (`PROGRESS.md`) — hasil lama sudah dihapus dan tak lagi ada di repo,
jadi Tahap 1 di atas menguji ulang dari nol, bukan mengasumsikan temuan itu berulang.

Prior-work tugas identik (deteksi judi Indonesia) untuk pembanding hasil akhir:
Kamdan dkk. (2025), Manullang dkk. (2025, JAIC) — daftar lengkap di `docs/DAFTAR_REFERENSI.pdf`.
