# 🫀 Heart Disease Prediction Using Ensemble Machine Learning Methods

**DSC 680 — Applied Data Science | Bellevue University | Spring 2026**

A supervised machine learning project analyzing the Cleveland Heart Disease dataset to predict the presence of heart disease using clinical intake features. The project compares five classification algorithms and recommends Logistic Regression as the primary clinical screening model based on AUC and recall performance.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methods](#methods)
- [Results](#results)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [References](#references)

---

## Project Overview

**Business Problem:** Cardiovascular disease is the leading cause of death in the United States, yet early-stage detection often requires costly and invasive diagnostic testing. This project explores whether a machine learning model trained on routine, non-invasive clinical measurements can reliably predict heart disease presence — supporting earlier, lower-cost screening decisions.

**Research Questions:**
1. Which clinical features are most predictive of heart disease presence?
2. Which classification algorithm produces the best balance of sensitivity and overall predictive performance?
3. How accurately can a model distinguish disease presence from absence using only intake-level measurements?

---

## Dataset

**Source:** [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

**Origin:** Cleveland Clinic Foundation (Detrano et al., 1989)

| Property | Value |
|---|---|
| Records | 303 (post-deduplication) |
| Features | 13 predictor variables + 1 target |
| Target | Binary (0 = no disease, 1 = disease present) |
| Class Balance | 54.1% negative / 45.9% positive |
| Missing Values | 6 rows (`ca`: 4, `thal`: 2) — imputed with median |

### Feature Summary

| Feature | Type | Description |
|---|---|---|
| `age` | Continuous | Patient age in years |
| `sex` | Binary | 0 = Female, 1 = Male |
| `cp` | Categorical | Chest pain type (1–4) |
| `trestbps` | Continuous | Resting blood pressure (mm Hg) |
| `chol` | Continuous | Serum cholesterol (mg/dl) |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dl |
| `restecg` | Categorical | Resting ECG results (0–2) |
| `thalach` | Continuous | Maximum heart rate achieved |
| `exang` | Binary | Exercise-induced angina |
| `oldpeak` | Continuous | ST depression (exercise vs. rest) |
| `slope` | Categorical | Slope of peak exercise ST segment |
| `ca` | Ordinal | Major vessels colored by fluoroscopy (0–3) |
| `thal` | Categorical | Thalassemia result (3=Normal, 6=Fixed, 7=Reversible) |
| `target` | Binary | Heart disease diagnosis (0=Absent, 1=Present) |

> **Note:** The original dataset contains 76 attributes; all published ML experiments use this 14-feature subset. The target was originally encoded 0–4 (severity); values 1–4 are collapsed to 1 (presence) per established convention.

---

## Repository Structure

```
Heart_Disease_Analysis/
│
├── data/
│   └── heart_disease_targets.csv        # Original Cleveland dataset
│
├── notebooks/
│   └── heart_disease_analysis.ipynb     # Full analysis pipeline
│
├── outputs/
│   ├── fig1_continuous.png              # Continuous feature distributions
│   ├── fig2_categorical.png             # Categorical feature distributions
│   ├── fig3_correlation.png             # Correlation heatmap
│   ├── fig5_model_comparison.png        # Cross-validated model metrics
│   ├── fig6_roc_curves.png              # ROC curves
│   ├── fig7_confusion_matrices.png      # Confusion matrices (top 3)
│   └── fig8_feature_importance.png      # RF importances + LR coefficients
│
├── paper/
│   ├── heart_disease_whitepaper_draft.pdf   # White paper (PDF)
│   └── heart_disease_whitepaper_draft.docx  # White paper (Word)
│
└── README.md
```

---

## Methods

### Phase 1 — Exploratory Data Analysis
- Distribution analysis of continuous and categorical features stratified by outcome
- Pearson correlation heatmap across all features
- Class balance check and outlier identification
- Missing value detection and median imputation

### Phase 2 — Model Training and Comparison
Five algorithms evaluated via **5-fold stratified cross-validation**:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline; scale-sensitive; fully interpretable via coefficients |
| Random Forest | Ensemble tree method; feature importance output |
| Gradient Boosting | Sequential boosting; tuned via GridSearchCV |
| SVM | Support Vector Machine; scale-sensitive |
| KNN | Distance-based; scale-sensitive |

Scale-sensitive models are wrapped in a `sklearn.Pipeline` with `StandardScaler` to prevent data leakage across CV folds.

**Primary metric:** Recall (sensitivity) — false negatives carry greater clinical risk than false positives in cardiac screening.

### Phase 3 — Hyperparameter Tuning and Ensemble
- `GridSearchCV` with AUC-ROC as optimization metric for top 3 models
- Soft-voting ensemble combining tuned LR, RF, and GB models

---

## Results

### Cross-Validated Performance (5-Fold)

| Model | Accuracy | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|
| **Logistic Regression** | 0.832 | **0.791** | 0.813 | **0.912** |
| **Random Forest** | 0.822 | 0.770 | 0.799 | **0.912** |
| SVM | **0.838** | 0.784 | **0.815** | 0.898 |
| Ensemble (Tuned) | 0.835 | 0.776 | 0.811 | 0.908 |
| KNN | 0.822 | 0.777 | 0.800 | 0.882 |
| Gradient Boosting | 0.782 | 0.733 | 0.754 | 0.900 |

### Confusion Matrix — Logistic Regression (Best Recall)

```
                  Predicted
                  No Disease   Disease
Actual  No Disease    144          20
        Disease        29         110

Recall:      0.791  (79.1% of true disease cases correctly identified)
Specificity: 0.878
```

---

## Key Findings

**🏆 Recommended Model: Logistic Regression**
- Best recall (0.791) — minimizes missed disease cases
- Tied best AUC (0.912) with Random Forest
- Fully interpretable via coefficient magnitudes — critical for clinical communication

**📊 Top 5 Predictive Features** (consistent across RF and LR):
1. `thal` — Thalassemia result (reversible defect strongly associated with disease)
2. `ca` — Number of major vessels colored by fluoroscopy
3. `cp` — Chest pain type (asymptomatic = highest disease rate)
4. `thalach` — Maximum heart rate achieved
5. `oldpeak` — ST depression induced by exercise

**⚠️ Low-signal features:** `fbs` (fasting blood sugar) and `chol` (cholesterol) contribute negligibly to both models and may be safely deprioritized in future iterations.

**📌 Notable observations:**
- The ensemble did not outperform individual top models — common with small datasets (n=303)
- Gradient Boosting underperformed expectations, likely due to insufficient data for effective sequential error correction
- Class balance (54/46) was near-ideal; no resampling required

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Bizarroxela/Heart_Disease_Analysis.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

---

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.13
jupyter
```

Install all at once:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## References

- Centers for Disease Control and Prevention. (2024). *Heart disease facts.* https://www.cdc.gov/heartdisease/facts.htm
- Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology, 64*(5), 304–310. https://doi.org/10.1016/0002-9149(89)90524-9
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Siegel, E. (2016). *Predictive analytics: The power to predict who will click, buy, lie, or die* (Rev. ed.). Wiley. ISBN: 9781119145677
- UCI Machine Learning Repository. (n.d.). *Heart disease data set.* https://archive.ics.uci.edu/ml/datasets/heart+disease

---

*Submitted in partial fulfillment of the requirements for DSC 680 — Applied Data Science, Bellevue University, Spring 2026.*
