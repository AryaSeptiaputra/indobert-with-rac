# results/combined/

File turunan (bukan sumber kebenaran) hasil `python scripts/merge_runs.py`.
Menggabungkan CSV ringkasan run dari semua folder eksperimen menjadi tiga file:
`runs_rma.csv`, `runs_rmb.csv`, `runs_rmc.csv`.

Sumber yang digabung (nilai kolom `source`):

| `source` | folder asal |
|---|---|
| `vast` | `results/vast/` — campaign utama, angka final Bab 4 |
| `cheap_head` | `results/vast_rmc_cheap_head/` — eksplorasi RM-c dengan head RM-b termurah (linear/256) |
| `candidate2` | `results_c2/vast_candidate2/` — kandidat #2: config runner-up (RM-a wd=0.10, RM-b wd=0.01, RM-c α=0.2/k=3) |

Ketiganya memakai encoder yang sama (`indobenchmark/indobert-base-p2`); yang berbeda adalah
hyperparameter dan head, bukan encoder-nya.

## Cara membaca — baca ini sebelum menganalisis

- **Kunci baris adalah `(source, run_id)`, bukan `run_id` saja.** Tiap folder memulai penomoran
  dari 1, jadi `run_id` berulang antar-source. Penomoran ulang global sengaja tidak dilakukan agar
  tiap baris tetap bisa ditelusuri balik ke file & figure aslinya (`figures/rma_run13_curve.png`).
- **`delta_vs_best_f1_macro_pp` dan `is_tie_with_best` hanya valid per-`source`.** Keduanya dihitung
  `RunLogger.log` (`src/tuning.py:74`) relatif terhadap juara di dalam file asalnya. Jangan dipakai
  lintas-source, jangan dihitung ulang secara global.
- **Kolom efisiensi tidak boleh dibandingkan lintas-`source`.** Aturan validitas Bab 4: waktu latih,
  latensi, dan peak memory hanya sebanding bila diukur pada satu hardware + satu sesi. `source`
  berbeda = mesin/sesi berbeda. F1 tidak bergantung hardware, jadi aman dibandingkan.
- **Sel kosong = kolom itu memang tidak ada di file sumbernya**, bukan nilai hilang. Header antar
  campaign berbeda (mis. kolom `test_*` hanya ada di campaign yang membuka test set).
- `model_name` untuk `source == "vast"` diisi `indobenchmark/indobert-base-p2` oleh script
  (campaign itu berjalan sebelum kolom `model_name` ada; encoder-nya terdokumentasi di `CLAUDE.md`).

Angka final Bab 4 tetap dikutip dari `results/vast/`, bukan dari folder ini.

## Catatan data: `run_id` juara RM-c di `results/vast/best.json`

`best.json` mencatat juara RM-c sebagai `run_id: 13` dengan config `alpha=0.2, k=5, similarity`
dan `val_f1_macro=0.969903`. Di `runs_rmc.csv`, **config + F1 itu milik run #15**; run #13 justru
`k=1` dengan F1 0.969995. Jadi `run_id` di `best.json` bergeser, sementara **config dan F1-nya
benar** dan konsisten dengan yang dilaporkan di `CLAUDE.md`/Bab 4 (α=0.2, k=5, similarity).
Run #13 (`k=1`) memang sedikit lebih tinggi (+0,009 pp) tetapi sengaja tidak dipilih karena
`k=1` dinilai tidak stabil — lihat catatan di `run_candidate2_vast.sh`. Tidak diubah di sini:
`results/vast/` adalah angka Bab 4 yang sudah terkunci.
