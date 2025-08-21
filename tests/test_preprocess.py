import pandas as pd
from src import preprocess, features

def test_basic_and_features():
    df = pd.DataFrame({
        "income": [10000, None, 50000],
        "age": [23, 45, None],
        "employment_years": [0, 10, None],
        "num_credit_lines": [2, None, 4],
        "outstanding_debt": [5000, 0, None],
        "credit_utilisation": [0.3, None, 0.05],
        "late_payments": [0, 3, None],
        "delinquencies": [0, 1, None],
        "mortgage": [1, 0, None],
        "loan_amount": [2000, None, 5000],
        "default": [0,1,0]
    })
    clean = preprocess.basic_clean(df)
    feat = features.add_features(clean)
    assert "debt_to_income" in feat.columns
    assert feat["debt_to_income"].isnull().sum() == 0