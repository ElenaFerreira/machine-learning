# Big Data & Machine Learning — TPs M2TL

Elena FERREIRA - M2TL Digital Campus

## 🎯 Fil rouge

Prédiction du **churn client** sur le dataset public **Telco Customer Churn** (IBM). L'objectif métier : identifier en amont les clients d'un opérateur télécom susceptibles de résilier pour cibler une campagne de rétention.

Chaque TP couvre une étape du pipeline ML, de l'exploration des données à la modélisation, avec un transfert d'apprentissage progressif d'un TP à l'autre.

## 📁 Structure du repo

```
├── data/                       # Splits TP1 + splits featurisés TP3
│   ├── X_train.csv, X_val.csv, X_test.csv
│   ├── y_train.csv, y_val.csv, y_test.csv
│   └── X_train_fe.csv, X_val_fe.csv, X_test_fe.csv
├── artifacts/                  # Best model + manifeste (TP4)
│   ├── best_model.joblib
│   └── manifest.json
├── TP1.ipynb                   # EDA & preprocessing
├── TP2.ipynb                   # Baseline LogReg & métriques
├── TP3.ipynb                   # Pipeline pro & course de modèles (S2 matin)
├── TP4.ipynb                   # Tuning, seuil métier & best model (S2 après-midi)
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

## 📚 TP3 — Pipeline pro & course de modèles

**Notebook :** `TP3.ipynb`

Feature engineering, cross-validation stratifiée multi-métriques, tracking MLflow, et comparaison de 5 familles de modèles pour identifier les 2 candidats à tuner.

**Résultats clés :**

- 4 features ingénierées ajoutées : `tenure_group`, `services_count`, `has_internet`, `avg_charge_per_month`
- CV 5-fold stratifiée multi-métriques (F1, ROC-AUC, PR-AUC)
- Course de 5 modèles : LogReg (F1=0.5828), DecisionTree (0.5510), RandomForest (0.5628), GradientBoosting (0.5892), HistGradientBoosting (0.5646)
- Course très serrée en tête : tous les bons modèles convergent vers F1 ≈ 0.58, écart non significatif
- Tracking complet via MLflow (expérience `churn-models-zoo`)
- 2 candidats retenus pour tuning : LogReg (rapide, stable, interprétable) + HistGradientBoosting (potentiel via tuning)

## 📚 TP4 — Tuning, seuil métier & best model

**Notebook :** `TP4.ipynb`

GridSearchCV sur 2 modèles, optimisation du seuil selon une fonction de coût métier, choix raisonné du best model, permutation importance et évaluation finale (1 seule fois) sur le test set.

**Résultats clés :**

- LogReg tuned : F1 CV = 0.6308 (+0.048 vs default), best params `C=0.01, class_weight="balanced"`
- HGBT tuned : F1 CV = 0.6293 (+0.065 vs default), best params `lr=0.05, max_depth=5, max_iter=200, class_weight="balanced"`
- Insight clé : `class_weight="balanced"` est le levier dominant (+4.5 pts F1), mathématiquement équivalent à baisser le seuil
- Fonction de coût métier : gain = 55€ × TP - 5€ × FP (hypothèse 30% succès rétention, LTV 200€, appel 5€)
- Seuils optimaux : HGBT @ 0.15 (12 775€), LogReg @ 0.25 (12 675€) — bien sous le 0.5 par défaut
- **Best model retenu** : HistGradientBoosting tuned @ seuil 0.15
- Top 3 features (permutation importance) : `tenure` (0.042), `Contract` (0.020), `InternetService` (0.013)
- **Évaluation finale test set** : F1 0.5509, PR-AUC 0.6448, Recall 94.3%, **Gain métier 12 495 €** (recall 94 % = 265/281 churners détectés)
- Modèle persisté dans `artifacts/best_model.joblib` + manifeste JSON pour la session 4 (API FastAPI)

## 📚 TP5 — Segmentation client non-supervisée

**Notebook :** `TP5.ipynb`

Application de k-means et DBSCAN pour identifier des segments client sans utiliser la cible, visualisation PCA et profiling métier des clusters.

**Résultats clés :**

- Choix de K=4 (compromis elbow / silhouette / besoin métier — K=2 mathématiquement optimal mais trop pauvre côté business)
- 4 clusters équilibrés (22-29% chacun), PCA 2D capturant 52.6% de variance
- 4 personas identifiés et nommés :
  - **Newcomers Fiber à risque** (27%, churn 57%) — cible prioritaire rétention
  - **Téléphonie pure fidèles** (22%, churn 7%) — segment stable
  - **Loyalists Fiber premium** (29%, churn 14%) — à protéger via upsell
  - **Newcomers DSL standard** (22%, churn 26%) — levier qualité de service
- DBSCAN inadapté au dataset (1 cluster mastodonte à 77% + 1 cluster à 22%) — confirme empiriquement que DBSCAN brille sur données géométriques, pas tabulaires homogènes
- Convergence supervisé ↔ non-supervisé : le clustering retrouve les patterns du TP1 (effet de seuil tenure ≈ 25 mois, triplet à risque Month-to-month + Fiber + Electronic check)

## 📚 TP6 — Réinjection du cluster comme feature

**Notebook :** `TP6.ipynb`

Évaluation rigoureuse "avec vs sans" la feature `cluster_id` (CV 5-fold + test), test PCA en amont du modèle, décision Tech Lead sur l'adoption ou le rejet de la feature.

**Résultats clés :**

- Pipeline anti-fuite : k-means fitté sur train uniquement, prédiction sur val/test
- **CV 5-fold avec vs sans cluster** : ΔF1 = +0.0019 (3/5 folds négatifs, std des deltas = 0.0100 → non significatif), ΔPR-AUC = +0.0019
- **Test set** : ΔF1 = -0.0043, ΔPR-AUC = -0.0130 — signe inversé entre CV et test → confirmation du bruit
- **Décision : rejet de la feature cluster** (gain absent, variance accrue, dette technique non justifiée)
- Bonus PCA(10) en amont HGBT : F1 0.6293 → 0.6111 (-0.018) — PCA contre-productive sur modèles non-linéaires avec features déjà propres
- `kmeans.joblib` conservé pour la **segmentation marketing** (4 personas), pas pour le modèle prédictif
- `best_model.joblib` du TP4 reste le modèle de production
