# Instead of directly utilising a premade dataset, we will generate synthetic data
import argparse
import numpy as np
import pandas as pd

def generate(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.normal(40000, 20000, n).clip(8000, 200000).round(2)
    age = rng.integers(18, 75, n)
    employment_years = rng.integers(0, 40, n)
    num_credit_lines = rng.integers(0, 12, n)
    outstanding_debt = (rng.normal(10000, 15000, n) + 0.1*income).clip(0)
    credit_utilisation = (outstanding_debt / (income + 1)).clip(0, 5)  # ratio
    late_payments = rng.poisson(0.5, n)
    delinquencies = rng.binomial(1, 0.05, n) * rng.integers(1,5,n)
    mortgage = rng.binomial(1, 0.45, n)
    loan_amount = rng.normal(8000, 10000, n).clip(0)
    
    score = (
        0.00002 * outstanding_debt
        - 0.00001 * income
        + 0.2 * late_payments
        + 0.4 * delinquencies
        + 0.1 * (credit_utilisation * 100)
        - 0.02 * employment_years
        + 0.1 * (age < 25).astype(float)
    )
    
    prob = 1 / (1 + np.exp(-score))
    default = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({
        "income": income,
        "age": age,
        "employment_years": employment_years,
        "num_credit_lines": num_credit_lines,
        "outstanding_debt": outstanding_debt.round(2),
        "credit_utilisation": credit_utilisation.round(3),
        "late_payments": late_payments,
        "delinquencies": delinquencies,
        "mortgage": mortgage,
        "loan_amount": loan_amount.round(2),
        "default": default
    })
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--output", type=str, default="data/raw/credit_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate(args.n, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")

if __name__ == "__main__":
    main()