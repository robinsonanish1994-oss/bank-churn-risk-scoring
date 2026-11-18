"""Shared preprocessing and feature engineering for the churn project.

Used by both the training pipeline and the Streamlit app so that the
features a model sees at inference time are identical to training.
"""
import numpy as np
import pandas as pd

TARGET = "Exited"
DROP_COLS = ["Year", "CustomerId", "Surname"]          # identifiers / constants
RAW_NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
               "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CATEGORICAL = {"Geography": ["France", "Germany", "Spain"],
               "Gender": ["Female", "Male"]}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived behavioural / relationship-strength features."""
    out = df.copy()
    # Balance-to-Salary ratio: how much of a customer's income is parked here
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1.0)
    # Product density: products held per year of relationship
    out["ProductDensity"] = out["NumOfProducts"] / (out["Tenure"] + 1.0)
    # Engagement x product interaction: active AND multi-product
    out["EngagementProduct"] = out["IsActiveMember"] * out["NumOfProducts"]
    # Age-tenure interactions
    out["AgeTenure"] = out["Age"] * out["Tenure"]
    out["TenureAgeRatio"] = out["Tenure"] / out["Age"]
    # Simple flags
    out["ZeroBalance"] = (out["Balance"] == 0).astype(int)
    out["MultiProduct3Plus"] = (out["NumOfProducts"] >= 3).astype(int)
    return out


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals with a fixed column set (drop_first)."""
    out = df.copy()
    for col, levels in CATEGORICAL.items():
        for lvl in levels[1:]:
            out[f"{col}_{lvl}"] = (out[col] == lvl).astype(int)
        out = out.drop(columns=col)
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> model-ready feature matrix (no target)."""
    X = df.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df.columns])
    X = engineer(X)
    X = encode(X)
    return X[feature_names()]


def feature_names():
    eng = ["BalanceSalaryRatio", "ProductDensity", "EngagementProduct",
           "AgeTenure", "TenureAgeRatio", "ZeroBalance", "MultiProduct3Plus"]
    cats = [f"{c}_{l}" for c, lv in CATEGORICAL.items() for l in lv[1:]]
    return RAW_NUMERIC + eng + cats


SCALE_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
              "EstimatedSalary", "BalanceSalaryRatio", "ProductDensity",
              "EngagementProduct", "AgeTenure", "TenureAgeRatio"]
