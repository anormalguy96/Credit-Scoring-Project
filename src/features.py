import pandas as pd, numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["debt_to_income"] = (df["outstanding_debt"] / (df["income"] + 1)).round(4)
    
    df["lines_per_year"] = df.apply(
        lambda r: r["num_credit_lines"] / (r["employment_years"] + 1), axis=1
    ).round(3)
    
    df["is_young"] = (df["age"] < 25).astype(int)
    
    df["high_utilisation"] = (df["credit_utilisation"] > 0.5).astype(int)
    
    df["recent_arrears"] = (df["late_payments"] >= 2).astype(int)
    
    df["mortgage_over_income"] = df["mortgage"] * (df["loan_amount"] / (df["income"] + 1))
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df