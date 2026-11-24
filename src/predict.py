"""Score customers with the saved model."""
import os, sys
import joblib, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from features import add_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = [(0.10, "Low"), (0.20, "Moderate"), (0.35, "Elevated"), (1.01, "High")]


def load_model():
    return joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))


def risk_band(pd_):
    return next(label for cut, label in BANDS if pd_ < cut)


def score(records: pd.DataFrame, bundle=None) -> pd.DataFrame:
    """records: raw dataset columns (PAY_1 naming). Returns PD, band and decision flag."""
    bundle = bundle or load_model()
    X = add_features(records)[bundle["columns"]]
    p = bundle["model"].predict_proba(X)[:, 1]
    return pd.DataFrame({"pd": p, "risk_band": [risk_band(x) for x in p], "flag_high_risk": p >= bundle["threshold"]},
                        index=records.index)


if __name__ == "__main__":
    from features import load_clean
    df = load_clean(os.path.join(ROOT, "data", "UCI_Credit_Card.csv")).head(5)
    print(pd.concat([df[["ID", "LIMIT_BAL", "PAY_1", "DEFAULT"]], score(df)], axis=1).round(3).to_string(index=False))
