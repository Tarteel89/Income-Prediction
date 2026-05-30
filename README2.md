# Income Prediction — U.S. Census Classification

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A two-part machine learning project predicting whether a U.S. adult earns above or below $50K/year, using the UCI Adult Census dataset. Covers the full ML pipeline: cleaning, EDA, SMOTE, PCA, KMeans feature engineering, embedded feature selection, and permutation importance.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Part 2 — Feature Engineering & Selection](#part-2--feature-engineering--selection)
- [Results Summary](#results-summary)
- [Tech Stack](#tech-stack)
- [Key Insights](#key-insights)
- [Author](#author)

---

## Overview

This project tackles a supervised binary classification problem on the UCI Adult Census dataset. The goal is to identify which demographic and employment factors best predict whether an individual earns more than $50,000 per year — a question relevant to workforce analytics, policy research, and socioeconomic studies.

The work is split across two notebooks:
- **Part 1** establishes a baseline Random Forest model with full preprocessing, EDA, SMOTE balancing, and PCA experiments.
- **Part 2** extends the pipeline with engineered features (PCA components + KMeans cluster labels), embedded feature selection, and a leaner final model.

---


## Part 2 — Feature Engineering & Selection

| Step | Description |
|---|---|
| 1. Feature Engineering | Concatenate **3 PCA components** + **KMeans cluster label** with original features |
| 2. Cluster Selection | Elbow curve to choose optimal `k` |
| 3. Engineered Model | Random Forest on the expanded feature set |
| 4. Feature Selection | `SelectFromModel` (embedded method) with mean importance threshold |
| 5. Final Model | Random Forest on selected features only |
| 6. Permutation Importance | Top 10 features on final model + commentary vs Part 1 |

> All transformations (PCA, KMeans, SMOTE, scaling) are **fit on training data only** and applied to test data via `.transform()` / `.predict()` to prevent data leakage.

---

## Results Summary

| Model | # Features | Test Accuracy | Train Time |
|---|---|---|---|
| Part 1 — Baseline | 80 | 83.83% | 17.30s |
| Part 2 — Engineered Features | 84 | 83.69% | 11.09s |
| **Part 2 — Final Selected** | **13** | **82.99%** | **7.39s** |

The final model uses **84% fewer features** than the baseline and trains in **less than half the time**, with only a 0.8 percentage point drop in accuracy — a strong simplicity-vs-performance tradeoff.

---

## Key Insights

### Top Predictors (Permutation Importance — Final Model)

| Rank | Feature | Note |
|---|---|---|
| 1 | `capital-gain` | Strongest single predictor — signals investment income |
| 2 | `educational-num` | More education = higher earning potential |
| 3 | `marital-status_Married-civ-spouse` | Life stage correlation with peak earning years |
| 4 | `age` | Seniority and career stage |
| 5 | `capital-loss` | Investment activity signal |
| 6 | `occupation_Exec-managerial` | High-earning role indicator |
| 7 | **`cluster`** *(engineered)* | KMeans group membership added new signal |
| 8 | `gender_Male` | Demographic correlate |
| 9 | `hours-per-week` | Work intensity |
| 10 | `occupation_Prof-specialty` | Professional roles |

### Did Feature Engineering Help?

- **KMeans clustering succeeded:** `cluster` made it into the top 10 (rank 7), outranking `gender_Male`, `hours-per-week`, and `occupation_Prof-specialty`.
- **PCA did not help here:** `PC1`, `PC2`, `PC3` did not pass the `SelectFromModel` threshold — the original features already captured the linear signal PCA would extract.

### Stakeholder Insights from Part 1

- **Education:** The share of adults earning >$50K rises monotonically from under 2% (early school leavers) to over 70% (doctorate holders). A bachelor''s degree is the inflection point.
- **Age:** High-earner rates follow a classic career arc, peaking at ~40% in the **45-54 age band**, then declining toward retirement.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Modelling, scaling, PCA, KMeans, feature selection, evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `Jupyter Notebook` | Interactive development |

---

## Author

**Tarteel Abu Ghazaleh** — [@Tarteel89](https://github.com/Tarteel89)

> *Always learning, always improving.*
