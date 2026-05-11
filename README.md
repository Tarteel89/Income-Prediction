# Income Prediction — U.S. Census Classification

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Predicting whether a U.S. adult earns above or below $50K/year using demographic and employment data from the UCI Adult Census dataset.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Key Insights](#key-insights)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [File Structure](#file-structure)

---

## Overview

This is a supervised binary classification project built on the UCI Adult Census dataset. The goal is to identify the demographic and employment factors that best predict whether an individual earns more than $50,000 per year.

The project covers the full ML pipeline: data cleaning, exploratory analysis, class imbalance handling with SMOTE, dimensionality reduction with PCA, feature importance via permutation importance, and stakeholder-ready visualisations.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult) |
| Rows | 48,842 |
| Features | 14 (6 numeric, 8 categorical) |
| Target | `income`: `>50K` or `<=50K` |
| Class balance | 76% `<=50K` / 24% `>50K` (imbalanced) |

**One row = one census respondent.**

---

## Project Workflow

| Step | Description |
|---|---|
| 1. Setup | Helper functions, display config, sklearn pandas output |
| 2. Load & Inspect | `head()`, `dtypes`, null checks, `?` placeholder detection |
| 3. Cleaning | Replace `?` placeholders, drop rows, strip whitespace |
| 4. EDA | Class distribution, feature histograms, income rates by category |
| 5. Preprocessing | Drop redundant columns, one-hot encode, `StandardScaler` |
| 6. SMOTE | Oversample minority class on training data only |
| 7. Baseline Model | `RandomForestClassifier` (100 trees), timed |
| 8. PCA Experiments | `PCA(3)` → variance curve → `PCA(0.85)`, speed comparison |
| 9. Permutation Importance | Top 10 features ranked by mean accuracy drop |
| 10. Explanatory Visuals | Two stakeholder-ready charts with written insights |

---

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| RF Baseline | ~86% | Full feature set after SMOTE |
| RF + PCA(3) | Lower | Only 3 components — too compressed |
| RF + PCA(0.85) | Comparable | Faster training, retains 85% variance |

> PCA(0.85) achieves similar accuracy to the baseline with meaningfully faster training — a worthwhile tradeoff at scale.

---

## Key Insights

### Top Predictors (Permutation Importance)

| Rank | Feature | Why It Matters |
|---|---|---|
| 1 | `educational-num` | More education = higher earning potential |
| 2 | `age` | Seniority and career stage |
| 3 | `capital-gain` | Signals investment income |
| 4 | `hours-per-week` | Full-time/overtime correlates with income |
| 5 | Marital status dummies | Life stage correlation with peak earning years |

### Education
The share of adults earning >$50K rises monotonically from under 2% (early school leavers) to over 70% (doctorate holders). A **bachelor's degree** is the inflection point.

### Age
High-earner rates follow a classic career arc, **peaking at ~40% in the 45-54 age band**, then declining toward retirement.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Modelling, scaling, PCA, evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `Jupyter Notebook` | Interactive development |

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Tarteel89/adult-income-classification.git
cd adult-income-classification

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter

# 3. Run the notebook
jupyter notebook project_2_adult_income.ipynb
```

---

## File Structure

```
adult-income-classification/
├── adult.csv                        # Raw dataset
├── project_2_adult_income.ipynb     # Main analysis notebook
└── README.md
```

---

## Author

**Tarteel Abu Ghazaleh** — [@Tarteel89](https://github.com/Tarteel89)

> *Always learning, always improving.*
