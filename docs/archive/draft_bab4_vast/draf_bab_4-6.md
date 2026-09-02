# 4.6 Evaluasi Kriteria Sukses

Penelitian ini menetapkan tiga kriteria kuantitatif untuk menilai apakah strategi adaptasi
ringan (RM-b dan RM-c) dapat dikatakan **kompetitif** terhadap *full fine-tuning* (RM-a).
Sebuah strategi dinyatakan kompetitif jika memenuhi **minimal 2 dari 3** kriteria berikut:

1. Selisih F1-score (F1-macro) ≤ 3 poin persentase terhadap RM-a
2. Reduksi *trainable parameters* ≥ 90%
3. Reduksi waktu latih ≥ 50%

## 4.6.1 Hasil Evaluasi

| Kriteria | RM-b | RM-c |
|---|---|---|
| Gap F1-macro (test) | 1,21pp ✅ (≤3pp) | 1,10pp ✅ (≤3pp) |
| Reduksi *trainable parameters* | 99,28% ✅ (≥90%) | 100% ✅ (≥90%) |
| Reduksi waktu latih | 85,64% ✅ (≥50%) | 100% ✅ (≥50%) |
| **Kriteria terpenuhi** | **3/3** | **3/3** |
| **Status** | **Kompetitif** | **Kompetitif** |

Tabel 4.12 Verdict kriteria sukses (`results/vast/metrics/success_criteria.csv`)

## 4.6.2 Pembahasan

Baik RM-b maupun RM-c **melampaui syarat minimal** (2 dari 3 kriteria) dengan memenuhi
**seluruh tiga kriteria sekaligus** — sebuah verdict kuantitatif dan obyektif, bukan
penilaian kualitatif, bahwa kedua strategi *frozen-encoder* kompetitif terhadap *full
fine-tuning* menurut definisi yang ditetapkan di awal penelitian ini. Konsisten dengan
temuan §4.5, kriteria yang dipenuhi dengan margin paling besar adalah reduksi parameter dan
reduksi waktu latih (kedua strategi jauh melampaui ambang 90%/50%), sementara gap F1-macro
— meski tetap lolos dengan margin nyaman (≤1,21pp dari batas 3pp) — merupakan kriteria
dengan margin kelonggaran paling tipis di antara ketiganya.

Penting dicatat bahwa ketiga kriteria di atas **tidak mencakup *latency* inferensi** — pada
metrik ini, sebagaimana dibahas di §4.5.3, RM-b/RM-c tidak menunjukkan perbaikan berarti
dibanding RM-a. Verdict "kompetitif" pada subbab ini karena itu perlu dibaca dengan konteks
bahwa efisiensi yang terukur adalah efisiensi *pelatihan*, bukan efisiensi *inferensi* —
nuansa yang disintesis lebih lanjut pada §4.7.
