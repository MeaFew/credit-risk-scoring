# Documentation

Extended documentation for the riskscore credit-risk scoring project.

## Contents

- **Methodology** — deeper notes on WOE/IV binning, the time-aware validation
  strategy, and the rationale for the model stack (LR → RF → XGBoost → LightGBM).
- **SHAP interpretation** — narrative walkthroughs of the dependence plots and
  global feature importance in `../images/`.

The project README (`../README.md`) covers quick start, the tech-stack table,
and the benchmark / model-results tables. The Home Credit column dictionary
ships at `../data/raw/HomeCredit_columns_description.csv`.

---

## Methodology

### 1. Data and problem framing

The project is built around the Kaggle *Home Credit Default Risk* competition.
The goal is to predict the probability that a loan applicant will default on a
consumer credit product. The raw training set contains `application_train.csv`
with a binary `TARGET` column (1 = default, 0 = repaid), while the official test
set `application_test.csv` is delivered **without** labels. This means the only
reliable way to estimate generalisation is cross-validation on the training set,
not a held-out test AUC.

### 2. Preprocessing

`scripts/preprocess.py` performs the following steps, with leakage prevention as
a first-class concern:

1. **Target separation**: `TARGET` is removed from the train frame before any
   statistic is computed.
2. **Column alignment**: only columns shared between train and test are kept.
3. **Outlier capping**: per-column 0.1st and 99.9th percentiles are computed on
   the train set only (`compute_outlier_bounds`), then applied to both train and
   test via `cap_outliers_with_bounds`.
4. **Missing-value imputation**: a `SimpleImputer(strategy="median")` is fit on
   the train set only and then transforms both frames.
5. **Categorical fill**: missing categoricals are filled with the literal string
   `"MISSING"`, which is target-independent and therefore safe to apply globally.

The output is written to `data/processed/application_train_cleaned.csv` and
`application_test_cleaned.csv`.

### 3. Feature engineering

`scripts/feature_engineering.py` builds a deterministic, target-free feature
matrix. The philosophy is that any transformation that uses the target must live
inside the cross-validation loop; everything else can be pre-computed once.

Generated feature groups:

- **Ratio features**: `CREDIT_TO_INCOME_RATIO`, `ANNUITY_TO_INCOME_RATIO`,
  `CREDIT_TO_ANNUITY_RATIO`, `EMPLOYED_TO_AGE_RATIO`, and `AGE_YEARS` from
  `DAYS_BIRTH`.
- **EXT_SOURCE interactions**: pairwise products and squares of the three
  external credit score columns. These are the strongest predictors in the
  dataset.
- **Low-cardinality categoricals**: one-hot encoded with `drop_first=True` after
  concatenating train and test so both frames share the same dummy columns.
- **High-cardinality categoricals**: kept as raw strings. They are target-encoded
  **inside** the model CV pipeline, never here.

#### WOE / IV (analysis only)

`compute_woe_iv` quantile-bins each numeric feature and computes Weight of
Evidence (WOE) and Information Value (IV):

```
WOE = ln( good_dist / bad_dist )
IV  = Σ (good_dist - bad_dist) * WOE
```

A small Laplace-style smoothing (`+0.5`) avoids division-by-zero on sparse bins.
The resulting `iv_report.csv` is for documentation and sanity checking only; it
is **not** used to filter features before modeling, because selecting features by
IV estimated from the full training target would leak target information into
each CV fold.

### 4. Target encoding without leakage

High-cardinality categorical columns (e.g. `OCCUPATION_TYPE`,
`ORGANIZATION_TYPE`) are encoded with `sklearn.preprocessing.TargetEncoder`.
The encoder is wrapped as the first step of a sklearn `Pipeline` in
`scripts/train_models.py`:

```python
Pipeline([
    ("preprocess", ColumnTransformer([("te", TargetEncoder(target_type="binary"), cat_cols)],
                                      remainder="passthrough")),
    ("model", classifier),
])
```

A fresh pipeline is created for every CV fold and fit on the fold's training
split only. Consequently, validation rows never contribute their own target to
their encoded value — this removes the most common source of leakage in credit
scoring pipelines.

### 5. Model stack and validation

Four models are compared:

| Model | Role |
|-------|------|
| Logistic Regression | Linear baseline; highly interpretable. |
| Random Forest | Non-linear ensemble; robust to outliers. |
| XGBoost | Industry-standard gradient boosting with histogram tree method. |
| LightGBM | Faster gradient boosting with leaf-wise tree growth. |

All models are evaluated with 5-fold stratified cross-validation using AUC, KS,
and Gini. The best model by mean CV AUC is retrained on the full training set
and persisted as a joblib pipeline (`models/xgb_risk_model.joblib`).

Because the official test set has no labels, the project reports two honest
estimates of generalisation:

1. **5-fold CV metrics** from `train_models.py`.
2. **Out-of-fold (OOF) metrics** from `evaluate.py`: every training row is
   scored by a model that did not see it during fitting. The OOF vector is saved
   to `reports/oof_predictions.npy` and used by the Streamlit dashboard's
   *Score Distribution* page.

### 6. Threshold and business interpretation

`evaluate.py` picks the threshold that maximises Youden's J statistic
(`tpr - fpr`). The resulting confusion matrix is reported in `oof_metrics` and
exposed in the dashboard. In production, the threshold should be tuned against
the lender's cost of false positives vs. false negatives rather than a pure
statistical criterion.

---

## SHAP interpretation

`scripts/shap_analysis.py` explains the best model with SHAP (SHapley Additive
exPlanations). Tree-based models use `shap.TreeExplainer`; the logistic
regression path would use `shap.LinearExplainer`.

### 1. Global importance — summary plot

`images/shap_summary.png` ranks features by the mean absolute SHAP value. On the
Home Credit data the top drivers are consistently:

- `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` and their pairwise products.
- `CREDIT_TO_INCOME_RATIO` and `ANNUITY_TO_INCOME_RATIO`.
- `AGE_YEARS` / `DAYS_BIRTH`.

A feature can have high importance either because it strongly pushes predictions
in one direction (e.g. low external score → high default risk) or because it
pushes in both directions for different sub-populations (e.g. young vs. old
applicants).

### 2. Dependence plots

`images/shap_dependence_*.png` show how the SHAP value of a single feature
changes with its own value, coloured by an interacting feature. Typical
patterns:

- **EXT_SOURCE_2**: lower values strongly increase default probability; the
  curve is monotonic and steep, reflecting that this bureau score is a direct
  default-risk signal.
- **CREDIT_TO_INCOME_RATIO**: very high ratios push predictions upward, but the
  effect plateaus as the ratio becomes extreme (outlier capping reduces
  sensitivity).
- **AGE_YEARS**: younger applicants tend to receive higher SHAP values unless
  offset by strong external credit scores; the colouring by `EXT_SOURCE_2`
  usually reveals this interaction.

### 3. Individual explanations — force plot

`reports/shap_force_plot.html` shows how each feature pushes a single
application's predicted probability away from the base rate toward the final
score. Red arrows increase default risk; blue arrows decrease it. This is the
artefact most useful for underwriting reviewers who need to justify a reject or
manual-review decision.

### 4. Interpreting the dashboard risk calculator

The *Risk Calculator* page reloads the persisted pipeline and predicts the
default probability for a user-selected test case. Because the pipeline contains
the fitted `TargetEncoder`, raw categorical strings are accepted directly. The
probability is coloured by risk band:

- **≤ 20%**: low risk — approve with standard terms.
- **20–50%**: medium risk — recommend manual review.
- **> 50%**: high risk — recommend rejection or additional collateral.

These bands are illustrative; a production deployment should calibrate them to
the portfolio's actual loss rates and approval volumes.
