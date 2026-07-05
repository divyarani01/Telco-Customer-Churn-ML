"""
Central configuration for the Customer Churn project.
All paths, constants, and model hyper-parameter search spaces live here.
Import this module; never hard-code values elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path
from scipy.stats import randint

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR        = Path(__file__).resolve().parents[2]
DATA_DIR        = ROOT_DIR / "data"
RAW_DATA_PATH   = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR      = ROOT_DIR / "models"
NOTEBOOKS_DIR   = ROOT_DIR / "notebooks"

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
RANDOM_STATE: int  = 42
CV_FOLDS: int      = 5
TEST_SIZE: float   = 0.2
MLFLOW_URI: str    = f"sqlite:///{ROOT_DIR / 'notebooks' / 'mlflow.db'}"
EXPERIMENT_NAME: str = "customer_churn"
REGISTRY_NAME: str   = "CustomerChurnBestModel"

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_COL: str = "Churn"

# ---------------------------------------------------------------------------
# Feature groups (raw, before any encoding)
# ---------------------------------------------------------------------------
DROP_COLS: list[str] = ["customerID"]

BINARY_COLUMNS: list[str] = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

BINARY_MAPPING: dict[str, int] = {
    "Yes": 1, "No": 0, "Male": 1, "Female": 0,
}

NOMINAL_COLUMNS: list[str] = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# ---------------------------------------------------------------------------
# Validation rules (Great Expectations)
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS: list[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

CATEGORY_RULES: dict[str, list] = {
    "gender":           ["Female", "Male"],
    "SeniorCitizen":    [0, 1],
    "Partner":          ["Yes", "No"],
    "Dependents":       ["Yes", "No"],
    "PhoneService":     ["Yes", "No"],
    "MultipleLines":    ["No", "Yes", "No phone service"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["No", "Yes", "No internet service"],
    "OnlineBackup":     ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport":      ["No", "Yes", "No internet service"],
    "StreamingTV":      ["No", "Yes", "No internet service"],
    "StreamingMovies":  ["No", "Yes", "No internet service"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
    "Churn": ["No", "Yes"],
}

NUMERICAL_RULES: dict[str, dict] = {
    "tenure":         {"min": 0, "max": 72},
    "MonthlyCharges": {"min": 0, "max": None},
    "TotalCharges":   {"min": 0, "max": None},
}

NULL_RULES: dict[str, bool] = {
    "customerID": False, "gender": False, "SeniorCitizen": False,
    "Partner": False, "Dependents": False, "tenure": False,
    "PhoneService": False, "MultipleLines": False, "InternetService": False,
    "OnlineSecurity": False, "OnlineBackup": False, "DeviceProtection": False,
    "TechSupport": False, "StreamingTV": False, "StreamingMovies": False,
    "Contract": False, "PaperlessBilling": False, "PaymentMethod": False,
    "TotalCharges": True,   # 11 blanks become NaN after pd.to_numeric
    "MonthlyCharges": False, "Churn": False,
}

UNIQUE_RULES: dict[str, bool] = {"customerID": True}

DTYPE_RULES: dict[str, str] = {
    "customerID": "object", "gender": "object", "SeniorCitizen": "int64",
    "Partner": "object", "Dependents": "object", "tenure": "int64",
    "PhoneService": "object", "MultipleLines": "object",
    "InternetService": "object", "OnlineSecurity": "object",
    "OnlineBackup": "object", "DeviceProtection": "object",
    "TechSupport": "object", "StreamingTV": "object",
    "StreamingMovies": "object", "Contract": "object",
    "PaperlessBilling": "object", "PaymentMethod": "object",
    "MonthlyCharges": "float64", "TotalCharges": "float64", "Churn": "object",
}

# ---------------------------------------------------------------------------
# Hyper-parameter search spaces  (exact values from the notebook)
# ---------------------------------------------------------------------------
PARAM_GRIDS: dict[str, dict] = {
    "Ridge LR (L2)": {
        "C":        [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        "solver":   ["lbfgs", "saga"],
        "max_iter": [500, 1000, 2000],
    },
    "Lasso LR (L1)": {
        "C":        [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        "solver":   ["liblinear", "saga"],
        "max_iter": [500, 1000, 2000],
    },
    "Elastic Net LR": {
        "C":        [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100],
        "l1_ratio": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        "max_iter": [1000, 2000, 3000],
    },
    "Decision Tree": {
        "max_depth":         [3, 4, 5, 6, 7, 8, 10, None],
        "min_samples_split": randint(2, 50),
        "min_samples_leaf":  randint(1, 30),
        "criterion":         ["gini", "entropy"],
        "class_weight":      ["balanced", None],
    },
    "Random Forest": {
        "n_estimators":      [100, 200, 300, 400, 500],
        "max_depth":         [5, 7, 10, 12, 15, None],
        "max_features":      ["sqrt", "log2", 0.5],
        "min_samples_leaf":  randint(1, 20),
        "min_samples_split": randint(2, 30),
        "bootstrap":         [True, False],
    },
    "XGBoost": {
        "n_estimators":     [200, 300, 400, 500],
        "max_depth":        [3, 4, 5, 6, 7],
        "learning_rate":    [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha":        [0, 0.01, 0.1, 0.5, 1],
        "reg_lambda":       [1, 1.5, 2, 3, 5],
        "min_child_weight": [1, 3, 5, 7],
    },
    "LightGBM": {
        "n_estimators":      [200, 300, 400, 500],
        "num_leaves":        [20, 31, 50, 70, 100],
        "max_depth":         [-1, 5, 7, 10],
        "learning_rate":     [0.01, 0.05, 0.1, 0.15],
        "subsample":         [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.9],
        "reg_alpha":         [0, 0.01, 0.1, 0.5],
        "reg_lambda":        [0, 0.01, 0.1, 1],
        "min_child_samples": [10, 20, 30, 50],
    },
    "CatBoost": {
        "iterations":           [200, 300, 400, 500],
        "depth":                [4, 5, 6, 7, 8],
        "learning_rate":        [0.01, 0.05, 0.1, 0.15],
        "l2_leaf_reg":          [1, 3, 5, 7, 10],
        "border_count":         [32, 64, 128],
        "bagging_temperature":  [0, 0.5, 1.0],
    },
}
