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
├── TP1.ipynb       # TP1 — EDA & preprocessing initial
├── TP2.ipynb       # TP2 — Baseline LogReg & métriques
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

## 📚 TP2 — Baseline Logistic Regression & Métriques

**Notebook :** `TP2.ipynb`

Construction d'un baseline de classification (régression logistique) avec pipeline scikit-learn complet (preprocessing + modèle), évaluation multi-métriques et interprétation des coefficients.

**Résultats clés :**

- Baseline LogReg @ seuil 0.5 : Accuracy 0.81, F1 0.62, ROC-AUC 0.85, PR-AUC 0.63 (vs Dummy 0.73)
- Top features confirmant l'EDA : `tenure` (−1.32), `Contract`, `InternetService`
- Modèle bien calibré (Brier 0.137, −30 % vs baseline aléatoire)
- Seuil F1-optimal : 0.286 (gain marginal +0.01 vs seuil 0.5)
- Anomalies détectées à corriger en TP3 : multicolinéarité `TotalCharges`/`tenure`/`MonthlyCharges`, 6 features redondantes `"No internet service"`
