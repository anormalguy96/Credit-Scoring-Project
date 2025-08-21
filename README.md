## Credit Scoring Model

This project predicts whether an individual is likely to default on a loan based on their financial history. It uses **classification models** such as Logistic Regression and Random Forest, combined with feature engineering and evaluation metrics that are standard in credit risk analysis.

The aim is to provide a clean, reproducible starting point for anyone working on credit scoring, risk modelling, or financial classification problems.

---

### Features

* **Synthetic dataset generator** → produces realistic financial data with variables such as income, debts, payment history, mortgage, etc.
* **Preprocessing pipeline** → cleans missing values, handles outliers, and prepares the dataset.
* **Feature engineering** → includes debt-to-income ratio, utilisation flags, arrears indicators, and more.
* **Models included**:

  * Logistic Regression (interpretable baseline)
  * Random Forest (strong, non-linear benchmark)
* **Evaluation metrics**: Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
* **Visual outputs**: Confusion Matrix and ROC Curve saved to `outputs/figs/`.

---

### Example Metrics

On the generated dataset, Random Forest typically achieves:

* **Precision**: \~0.74
* **Recall**: \~0.71
* **F1-score**: \~0.72
* **ROC-AUC**: \~0.80

*(Values vary depending on random seed and dataset size)*

---

### Next Steps

* Add hyperparameter tuning (GridSearchCV / RandomSearch).
* Try gradient boosting methods (XGBoost, LightGBM, CatBoost).
* Add explainability with SHAP or LIME.
* Incorporate fairness and bias checks.
* Prepare for deployment (Flask/FastAPI, Docker).

---

### Contributing

Feel free to fork, open issues, or suggest improvements. This project is designed as a **solid starting point** — you can extend it for coursework, research, or real-world applications.

---

### Licence

This project is under the MIT License. 
