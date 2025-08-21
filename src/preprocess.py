from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "income", "age", "employment_years", "num_credit_lines",
    "outstanding_debt", "credit_utilisation", "late_payments",
    "delinquencies", "mortgage", "loan_amount"
]
TARGET = "default"

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    
    for c in ["income", "outstanding_debt", "loan_amount"]:
        if c in df.columns:
            upper = df[c].quantile(0.995)
            df[c] = np.minimum(df[c], upper)
    
    for c in ["age",
              "employment_years",
              "num_credit_lines",
              "late_payments",
              "delinquencies",
              "mortgage"]:
        if c in df.columns:
            df[c] = df[c].round(0).astype(int)
    return df

def split(df, test_size=0.2, random_state=42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
