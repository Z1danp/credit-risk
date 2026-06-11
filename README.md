# Credit Risk Prediction — German Credit Dataset (My Second Learning Project 🔥)

Learning project machine learning kedua saya untuk memprediksi risiko kredit nasabah menggunakan dataset German Credit. Evaluasi model menggunakan **cost matrix berbasis business metric** yang mencerminkan konsekuensi asimetris antara false positive dan false negative dalam konteks perkreditan.

---

## Dataset

**Sumber:** [German Credit Data — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
**Dikurasi oleh:** Prof. Dr. Hans Hofmann, Universität Hamburg

| Keterangan | Detail |
|---|---|
| Jumlah sampel | 1.000 |
| Jumlah fitur | 20 (7 numerik, 13 kategorikal) |
| Target | 0 = Nasabah Baik, 1 = Nasabah Buruk |
| Proporsi kelas | 70% Baik / 30% Buruk |

---

## Struktur Repo

```
├── data/
│   └── german.data              # Dataset utama
├── docs/
│   └── german.doc               # Dokumentasi resmi dataset
├── notebook.ipynb               # Notebook utama
├── requirements.txt
└── README.md
```

---

## Metodologi

```
Load & Split Dataset
        ↓
Exploratory Data Analysis
├── Analisis univariat & distribusi target
├── Korelasi numerik (Point-Biserial)
└── Uji Chi-Square fitur kategorikal
        ↓
Feature Engineering
├── Standard Scaler (fitur numerik)
├── Ordinal Encoding (fitur ordinal)
├── One-Hot Encoding (fitur nominal) → Baseline LR & RF
└── Weight of Evidence / WoE (fitur kategorikal) → LR + WoE
        ↓
Modelling
├── Logistic Regression (Baseline → WoE → GridSearchCV)
└── Random Forest (Baseline → GridSearchCV)
        ↓
Evaluasi berbasis Cost Matrix
```

---

## Cost Matrix

Evaluasi menggunakan cost matrix resmi dari dokumentasi dataset:

|  | Predicted: Good | Predicted: Bad |
|---|---|---|
| **Actual: Good** | 0 | 1 |
| **Actual: Bad** | 5 | 0 |

Memprediksi nasabah buruk sebagai baik (False Negative) **5× lebih mahal** dibanding menolak nasabah baik (False Positive), mencerminkan kerugian langsung akibat gagal bayar.

$$\text{Total Cost} = (FP \times 1) + (FN \times 5)$$

---

## Hasil

Dievaluasi pada **test set (200 sampel)** yang dikunci sejak awal.

| Model | False Positive | False Negative | Total Cost | Accuracy |
|---|---|---|---|---|
| LR Baseline (OHE) | 47 | 17 | 132 | 0.68 |
| LR + WoE | 36 | 12 | **96** | **0.76** |
| Random Forest (tuned) | 50 | 16 | 130 | 0.67 |

**Model terbaik: Logistic Regression + WoE** dengan total cost **96**, sekaligus unggul di kedua jenis error (FP & FN) dibanding Random Forest.

Kontribusi terbesar penurunan cost berasal dari transisi ke WoE encoding (−36 poin, dari 132 ke 96), jauh melampaui kontribusi hyperparameter tuning — mengindikasikan bahwa kualitas representasi fitur lebih krusial dari kompleksitas model pada dataset ini.

> **Catatan metodologis.** Random Forest sempat mencatat rata-rata cost cross-validation lebih rendah (89.2 vs 100), namun tidak bertahan di test set (130). Ini indikasi overfitting terhadap fold tuning akibat grid hyperparameter yang luas (243 kombinasi vs 8 pada LR) dan signal-to-noise dataset yang rendah. Konsistensi CV↔test pada LR + WoE (100 → 96) menjadi dasar pemilihan model final.

---

## Cara Menjalankan

1. Clone repo ini

```bash
git clone https://github.com/Z1danp/credit-risk.git
cd credit-risk
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Jalankan notebook

```bash
jupyter notebook notebook.ipynb
```

---

## Dependencies

Lihat `requirements.txt`. Library utama yang digunakan:

- `pandas`, `numpy` — manipulasi data
- `scikit-learn` — modelling dan preprocessing
- `category_encoders` — WoE encoding
- `matplotlib`, `seaborn` — visualisasi
- `scipy` — uji statistik (Chi-Square, Point-Biserial)

---

## Referensi

- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository.
