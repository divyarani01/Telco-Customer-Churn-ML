"""
Standalone training script.

Trains all 8 models, saves each one to models/ via model_store,
and logs everything to MLflow.

Usage (from project root):
    & "C:\ProgramData\anaconda3\python.exe" scripts\train.py
"""
from __future__ import annotations

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from config.config import PARAM_GRIDS, RANDOM_STATE
from data.load_data import load_raw_data
from monitoring.mlflow_logger import setup_mlflow, log_mlflow_run, register_best_model
from pipelines.preprocessing_pipeline import build_preprocessing_pipeline
from pipelines.training_pipeline import run_model_workflow
from models.evaluate import build_leaderboard
from utils.logger import get_logger

logger = get_logger("train")

# ── Data ─────────────────────────────────────────────────────────────────────
df = load_raw_data()
X  = df.drop(columns=["Churn"])
y  = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
logger.info("Train: %d  Test: %d", len(X_train), len(X_test))

# ── Preprocessing pipelines (fit on train only) ───────────────────────────────
tree_pipe   = build_preprocessing_pipeline("tree")
linear_pipe = build_preprocessing_pipeline("linear")

X_tr_tree   = tree_pipe.fit_transform(X_train, y_train)
X_te_tree   = tree_pipe.transform(X_test)

X_tr_linear = linear_pipe.fit_transform(X_train, y_train)
X_te_linear = linear_pipe.transform(X_test)

feature_names = list(X_tr_tree.columns)

# Class-imbalance weight for XGBoost
scale_pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())

# ── Model definitions ─────────────────────────────────────────────────────────
MODELS = [
    # (name, estimator, X_tr, X_te, preprocessing_pipe, model_type)
    ("Ridge LR (L2)",
     LogisticRegression(penalty="l2", class_weight="balanced", random_state=RANDOM_STATE),
     X_tr_linear, X_te_linear, linear_pipe, "linear"),

    ("Lasso LR (L1)",
     LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE),
     X_tr_linear, X_te_linear, linear_pipe, "linear"),

    ("Elastic Net LR",
     LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, class_weight="balanced", random_state=RANDOM_STATE),
     X_tr_linear, X_te_linear, linear_pipe, "linear"),

    ("Decision Tree",
     DecisionTreeClassifier(random_state=RANDOM_STATE),
     X_tr_tree, X_te_tree, tree_pipe, "tree"),

    ("Random Forest",
     RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
     X_tr_tree, X_te_tree, tree_pipe, "tree"),

    ("XGBoost",
     XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0),
     X_tr_tree, X_te_tree, tree_pipe, "tree"),

    ("LightGBM",
     LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, verbose=-1),
     X_tr_tree, X_te_tree, tree_pipe, "tree"),

    ("CatBoost",
     CatBoostClassifier(auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=0),
     X_tr_tree, X_te_tree, tree_pipe, "tree"),
]

# ── Train ─────────────────────────────────────────────────────────────────────
setup_mlflow()
results_list = []
run_ids = {}

for name, estimator, X_tr, X_te, pipe, mtype in MODELS:
    result = run_model_workflow(
        name=name, model=estimator,
        param_dist=PARAM_GRIDS[name],
        X_tr=X_tr, y_tr=y_train,
        X_te=X_te, y_te=y_test,
        preprocessing_pipe=pipe,
        model_type=mtype,
        feature_names=feature_names,
    )
    results_list.append(result)

    # Log to MLflow
    run_id = log_mlflow_run(
        result=result, X_te=X_te, y_te=y_test,
        feature_names=feature_names,
    )
    run_ids[name] = run_id

# ── Leaderboard ───────────────────────────────────────────────────────────────
lb = build_leaderboard(results_list)
print("\n" + "=" * 70)
print("LEADERBOARD")
print("=" * 70)
print(lb.to_string())

# ── Register best model ───────────────────────────────────────────────────────
best_name = lb.iloc[0]["Model"]
register_best_model(run_ids[best_name], best_name)
logger.info("Best model: %s", best_name)
logger.info("All models saved to models/ — start the API with:")
logger.info("  docker compose up --build")
logger.info("  or: uvicorn src.api.main:app --reload")
