# Big Data & Machine Learning — TPs M2TL

Elena FERREIRA - M2TL Digital Campus

## 🎯 Fil rouge

Prédiction du **churn client** sur le dataset public **Telco Customer Churn** (IBM). L'objectif métier : identifier en amont les clients d'un opérateur télécom susceptibles de résilier pour cibler une campagne de rétention.

Chaque TP couvre une étape du pipeline ML, de l'exploration des données à la modélisation, avec un transfert d'apprentissage progressif d'un TP à l'autre.

## 📁 Structure du repo

```
├── data/       # Splits train val/test (générés au TP1)
│   ├── X_train.csv
│   ├── X_val.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_val.csv
│   └── y_test.csv
├── TP1.ipynb     # TP1 — EDA & preprocessing initial
├── .gitignore
└── README.md
```

## 📚 TP1 — EDA & Preprocessing initial

**Notebook :** `TP1.ipynb`

EDA du dataset Telco Customer Churn (7 043 clients × 21 colonnes) et split stratifié train/val/test (70/15/15).

**Résultats clés :**

- Taux de churn global : **26,54 %** (déséquilibre ~3:1)
- Top prédicteurs : `Contract`, `InternetService`, `PaymentMethod`, `tenure`
- Multicolinéarité détectée : `TotalCharges ≈ tenure × MonthlyCharges` (r = 0,9996)
- Splits sauvegardés dans `data/` pour réutilisation au TP2
