# Credit Scoring Project

Purpose:
Predict credit default (binary) using common financial features. The repository contains:
- Synthetic data generator
- Preprocessing and feature engineering
- Pipelines with logistic regression and random forest
- Model evaluation with Precision, Recall, F1, ROC-AUC, confusion matrix

Quickstart:
1. Install dependencies: pip install -r requirements.txt
2. Generate data: python scripts/generate_synthetic_data.py --output data/raw/credit_data.csv --n 20000
3. Train: python src/train.py --data data/raw/credit_data.csv --model outputs/models/credit_model.pkl

Notes:
- The synthetic generator is simplistic — replace with real data when available.
- For production, add: feature importance/explainability, calibration, fairness checks, robust cross-validation, logging, and CI.