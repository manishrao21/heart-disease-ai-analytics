# Real-World AI-Driven Analytics Solution Using Python

Predicting heart disease from patient health measurements using the UCI (Cleveland) Heart
Disease dataset. This repo contains our team's Google Colab notebooks and written reports
for both deliverables of the project:

- **Deliverable 1:** Data Preprocessing and Initial Model Development
- **Deliverable 2:** Model Optimization, Real-World Impact, and Ethical Evaluation

## Problem

Heart disease is one of the leading causes of death worldwide, and early detection helps
doctors act sooner. We frame this as a **binary classification** problem: given common
health measurements (age, blood pressure, cholesterol, chest pain type, max heart rate,
etc.), predict whether a patient is likely to have heart disease (`1`) or not (`0`).

## Dataset

- **Source:** [UCI Machine Learning Repository – Heart Disease (Cleveland)](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Size:** 303 patients, 13 input features + 1 diagnosis column
- Loaded directly from the UCI URL inside the notebook (no manual download needed)

## Deliverable 1 — what the notebook does

1. Setup and imports (pandas, NumPy, Matplotlib, Seaborn, scikit-learn)
2. Load and inspect the data (`shape`, `info`, `describe`, missing values)
3. EDA: target count plot, feature histograms, correlation heatmap
4. Cleaning: drop duplicates, median-fill missing `ca` / `thal`, binarize the target
5. Feature engineering: one-hot encoding + `StandardScaler`
6. Stratified train/test split (75/25)
7. Models: Logistic Regression and Random Forest
8. Evaluation: accuracy, ROC-AUC, classification report, confusion matrices
9. Hyperparameter tuning with `GridSearchCV`
10. Feature importance, Ethical Considerations, and Team Collaboration notes

## Results (test set)

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 84.2% | 0.927 |
| Random Forest | 84.2% | 0.94 |
| Random Forest (tuned) | 84.2% | — |

Best Random Forest params: `max_depth=5`, `n_estimators=100`.
Strongest predictors: chest pain type, number of major vessels (`ca`), max heart rate (`thalach`).

## Deliverable 2 — what the notebook does

Builds on Deliverable 1 to refine, evaluate, and ethically review the model:

1. Reuse the cleaning/preprocessing pipeline (scaler now fit on the training split only, to avoid leakage)
2. **Hyperparameter tuning:** wider `GridSearchCV` + `RandomizedSearchCV` on the Random Forest, plus Logistic Regression `C` tuning, all with 5-fold stratified CV scored on ROC-AUC
3. **Final model evaluation:** accuracy, precision, recall, F1, ROC-AUC, 5-fold cross-validation, confusion matrix, ROC curve, and precision-recall curve
4. **Fairness check:** model accuracy broken down by patient sex
5. Markdown sections for Ethical Considerations, Real-World Application, and Final Thoughts & Conclusion

### Deliverable 2 results (final tuned Random Forest)

| Metric | Value |
|--------|-------|
| Accuracy | 84.2% |
| Precision | 0.811 |
| Recall | 0.857 |
| F1-score | 0.833 |
| ROC-AUC | 0.945 |

- Best params: `max_depth=3`, `n_estimators=100`, `min_samples_split=5`, `max_features='sqrt'`
- 5-fold CV accuracy: ~81% (± ~0.05)
- Fairness check: Male ~82% vs Female ~89% accuracy (few female cases, so interpreted with caution)

## How to run

Open the notebook in Google Colab and run all cells (Runtime -> Run all). The dataset loads
from the UCI URL automatically, so no upload is required.

Or run locally:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook AI_Analytics_Heart_Disease_Deliverable1.ipynb
```

## Files

- `AI_Analytics_Heart_Disease_Deliverable1.ipynb` — Deliverable 1 Colab notebook (with outputs)
- `AI_Analytics_Heart_Disease_Report_Deliverable1.docx` — Deliverable 1 report
- `AI_Analytics_Heart_Disease_Deliverable2.ipynb` — Deliverable 2 Colab notebook (with outputs)
- `AI_Analytics_Heart_Disease_Report_Deliverable2.docx` — Deliverable 2 report

## Team

- Manish Rao Joginipalli — data loading, cleaning, and missing-value handling
- Manoj Kumar Kadiyala — exploratory data analysis and visualizations
- Sreyas Reddy Mallypally — modeling, evaluation, and hyperparameter tuning

## Ethical note

The Cleveland dataset is small and skewed toward male patients, so the model may not
generalize to all populations. Any real use would require checking performance across
demographic groups, protecting patient privacy, and keeping a human in the loop for
diagnosis.
