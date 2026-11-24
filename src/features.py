"""Data cleaning and feature engineering for the UCI credit-card default dataset."""
import numpy as np
import pandas as pd

TARGET = "DEFAULT"
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]
PAYSTAT = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# SEX and MARRIAGE are deliberately NOT used as model inputs (fair-lending practice).
PROTECTED = ["SEX", "MARRIAGE"]


def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"PAY_0": "PAY_1", "default.payment.next.month": TARGET})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})  # undocumented codes -> "other"
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lim = d["LIMIT_BAL"].replace(0, np.nan)
    d["AVG_UTILIZATION"] = (d[BILL].mean(axis=1) / lim).clip(-1, 3)
    d["LATEST_UTILIZATION"] = (d["BILL_AMT1"] / lim).clip(-1, 3)
    d["PAY_RATIO"] = (d[PAYAMT].sum(axis=1) / d[BILL].clip(lower=0).sum(axis=1).replace(0, np.nan)).fillna(1).clip(0, 5)
    d["MAX_DELAY"] = d[PAYSTAT].max(axis=1)
    d["MEAN_DELAY"] = d[PAYSTAT].mean(axis=1)
    d["N_LATE_MONTHS"] = (d[PAYSTAT] > 0).sum(axis=1)
    d["BILL_TREND"] = (d["BILL_AMT1"] - d["BILL_AMT6"]) / lim
    return d


def model_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ["ID", TARGET] + PROTECTED]
