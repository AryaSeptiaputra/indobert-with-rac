#!/bin/bash
# Jalankan di instance Vast.ai (bukan lokal) -- CLI langsung ke job_runner.py,
# tanpa perlu Streamlit UI. Lihat VAST_GUIDE.md untuk cara sewa/upload instance.
#
# Menghasilkan 3 kandidat #2 yang BELUM ada angka test-nya:
#   1. RM-A kandidat #2  (run#26 historis: sama dengan config resmi tapi weight_decay=0.10)
#   2. RM-B kandidat #2  (run#23 historis: sama dengan config resmi tapi weight_decay=0.01)
#   3. RM-C kandidat #2, kolom "head performa-terbaik" (alpha=0.2, k=3) -- butuh head
#      resmi run#28 lebih dulu, jadi step 3 di bawah mereproduksi head itu (deterministik,
#      seed=42, tak perlu transfer file checkpoint apa pun dari lokal).
#
# TIDAK menyentuh results/vast/ (folder hasil resmi Bab 4) -- semua ditulis ke out_dir
# baru: results/vast_candidate2/.
#
# Kandidat #2 kolom "head cost-termurah" (alpha=0.4, k=3) SUDAH selesai dijalankan lokal
# sebelumnya (RAC eval, bukan training berat) -- tidak perlu diulang di sini.

set -e
cd "$(dirname "$0")"

OUT=results/vast_candidate2
mkdir -p "$OUT"

echo "=== [1/4] RM-A kandidat #2 (run#26: weight_decay=0.10) ==="
cat > /tmp/job_rma_c2.json <<'EOF'
{
  "phase": "rma",
  "out_dir": "results/vast_candidate2",
  "smoke": false,
  "model_name": "indobenchmark/indobert-base-p2",
  "note": "Kandidat #2 RM-A (run#26 historis): sama dengan run#13 kecuali weight_decay=0.10 -- seri persis val F1 dengan kandidat #1",
  "eval_test": true,
  "config": {"lr": 0.00002, "epochs": 5, "batch": 32, "warmup_ratio": 0.1, "weight_decay": 0.10, "micro_batch": 32, "seed": 42}
}
EOF
python src/job_runner.py --config /tmp/job_rma_c2.json

echo ""
echo "=== [2/4] RM-B kandidat #2 (run#23: weight_decay=0.01) ==="
cat > /tmp/job_rmb_c2.json <<'EOF'
{
  "phase": "rmb",
  "out_dir": "results/vast_candidate2",
  "smoke": false,
  "model_name": "indobenchmark/indobert-base-p2",
  "note": "Kandidat #2 RM-B performa (run#23 historis): sama dengan run#28 kecuali weight_decay=0.01",
  "eval_test": true,
  "config": {"head_arch": "mlp", "hidden_dim": 1024, "epochs": 10, "lr": 0.001, "dropout": 0.1, "weight_decay": 0.01, "batch": 32, "seed": 42}
}
EOF
python src/job_runner.py --config /tmp/job_rmb_c2.json

echo ""
echo "=== [3/4] Reproduksi head resmi run#28 (dibutuhkan RM-C kandidat #2 di step 4) ==="
cat > /tmp/job_rmb_official.json <<'EOF'
{
  "phase": "rmb",
  "out_dir": "results/vast_candidate2",
  "smoke": false,
  "model_name": "indobenchmark/indobert-base-p2",
  "note": "Reproduksi head resmi run#28 (mlp/1024, wd=0.0) di out_dir ini -- jadi juara baru (val lebih tinggi dari kandidat #2 di atas), checkpoint-nya dipakai step 4",
  "eval_test": false,
  "config": {"head_arch": "mlp", "hidden_dim": 1024, "epochs": 10, "lr": 0.001, "dropout": 0.1, "weight_decay": 0.0, "batch": 32, "seed": 42}
}
EOF
python src/job_runner.py --config /tmp/job_rmb_official.json

echo ""
echo "=== [4/4] RM-C kandidat #2, kolom head performa-terbaik (alpha=0.2, k=3) ==="
cat > /tmp/job_rmc_c2.json <<'EOF'
{
  "phase": "rmc",
  "out_dir": "results/vast_candidate2",
  "smoke": false,
  "model_name": "indobenchmark/indobert-base-p2",
  "note": "Kandidat #2 RM-C (head performa-terbaik run#28): alpha=0.2/k=3 -- config stabil berikutnya setelah k=5 (kandidat #1), menghindari k=1 yang mekanis tertinggi tapi tak stabil",
  "eval_test": true,
  "config": {"alpha": 0.2, "k": 3, "weighting": "similarity"}
}
EOF
python src/job_runner.py --config /tmp/job_rmc_c2.json

echo ""
echo "=== SELESAI. Hasil di: $OUT ==="
echo "  runs_rma.csv -> baris RM-A kandidat #2 (val + test)"
echo "  runs_rmb.csv -> 2 baris baru: kandidat #2 (val+test) dan reproduksi run#28 (val saja, jadi checkpoint)"
echo "  runs_rmc.csv -> baris RM-C kandidat #2 kolom performa-terbaik (val + test)"
echo ""
echo "Kirim isi ketiga CSV (atau seluruh folder $OUT) balik ke Claude untuk dirangkum ke tabel perbandingan."
