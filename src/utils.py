import pandas as pd
from pathlib import Path

def ensure_path(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def save_df(df, path):
    p = ensure_path(path)
    df.to_csv(p, index=False)
    return p