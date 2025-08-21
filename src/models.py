from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import numpy as np

NUMERIC_FEATURES = [
    "income", "age", "employment_years", "num_credit_lines",
    "outstanding_debt", "credit_utilisation", "late_payments",
    "delinquencies", "mortgage", "loan_amount",
    
    "debt_to_income", "lines_per_year", "is_young",
    "high_utilisation", "recent_arrears", "mortgage_over_income"
]

CATEGORICAL_FEATURES = []

def make_pipeline(model_type="rf", random_state=42):
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ], remainder="drop"
    )

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", model)])
    return pipe

def cross_validate(pipe, X, y, cv=5):
    scores = {}
    scores["roc_auc"] = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    scores["f1"] = cross_val_score(pipe, X, y, cv=cv, scoring="f1", n_jobs=-1)
    return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}

def save_model(pipe, path):
    joblib.dump(pipe, path)