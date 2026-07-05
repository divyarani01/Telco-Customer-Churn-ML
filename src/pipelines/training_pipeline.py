"""run_model_workflow() — the generic training loop from notebook cell 57.

Every model (Ridge LR, Lasso LR, Elastic Net, Decision Tree, Random Forest,
XGBoost, LightGBM, CatBoost) runs through the same pipeline:

  1. Train baseline
  2. Tune (RandomizedSearchCV -> optional GridSearchCV)
  3. Evaluate at default threshold
  4. Plot CM / ROC / PR
  5. Threshold optimisation
  6. Final evaluation + classification report

Returns a result dict that can be appended to results_list and passed to
log_mlflow_run().
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from models.evaluate import (
    build_leaderboard,
    evaluate_model,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    threshold_optimization,
)
from models.train import train_model, tune_model
from config.config import CV_FOLDS, RANDOM_STATE
from utils.logger import get_logger
from sklearn.metrics import classification_report

logger = get_logger(__name__)


def run_model_workflow(
    name: str,
    model,
    param_dist: dict,
    X_tr,
    y_tr,
    X_te,
    y_te,
    n_iter: int = 30,
    cv: int = CV_FOLDS,
    refine_grid: dict | None = None,
    feature_names: list[str] | None = None,
) -> dict:
    """Full training-evaluation cycle for a single model.

    Mirrors notebook cell 57 run_model_workflow() exactly.
    Returns a result dict compatible with the leaderboard and MLflow logger.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)

    # 1. Baseline
    print("  [1/4] Training baseline ...")
    baseline, _ = train_model(model, X_tr, y_tr)
    base = evaluate_model(baseline, X_te, y_te)
    print(f"  Baseline  ROC-AUC: {base['roc_auc']}  F1: {base['f1']}")

    # 2. Tuning
    print("  [2/4] Tuning (RandomizedSearchCV) ...")
    best_model = tune_model(
        model, param_dist, X_tr, y_tr,
        n_iter=n_iter, cv=cv, refine_grid=refine_grid,
    )
    best_model, train_time = train_model(best_model, X_tr, y_tr)

    # 3. Evaluate at default threshold
    print("  [3/4] Evaluating ...")
    metrics = evaluate_model(best_model, X_te, y_te, threshold=0.5)
    print(f"  Tuned     ROC-AUC: {metrics['roc_auc']}  F1: {metrics['f1']}")

    # 4. 3-panel plot (CM / ROC / PR)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(name, fontsize=12, fontweight="bold")
    plot_confusion_matrix(y_te, metrics["y_pred"], name, ax=axes[0])
    plot_roc_curve(y_te, metrics["y_proba"], name, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve"); axes[1].legend(fontsize=7)
    plot_pr_curve(y_te, metrics["y_proba"], name, ax=axes[2])
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision-Recall Curve"); axes[2].legend(fontsize=7)
    plt.tight_layout()
    plt.show()

    # 5. Threshold optimisation
    print("  [4/4] Threshold optimisation ...")
    best_t = threshold_optimization(y_te, metrics["y_proba"])
    opt = evaluate_model(best_model, X_te, y_te, threshold=best_t)
    print(f"  Optimal threshold : {best_t}  ->  F1: {opt['f1']}")
    print(classification_report(y_te, opt["y_pred"], target_names=["No Churn", "Churn"]))

    return {
        "Model"          : name,
        "Accuracy"       : opt["accuracy"],
        "Precision"      : opt["precision"],
        "Recall"         : opt["recall"],
        "F1"             : opt["f1"],
        "ROC-AUC"        : opt["roc_auc"],
        "PR-AUC"         : opt["pr_auc"],
        "Best Threshold" : best_t,
        "Train Time (s)" : train_time,
        "Inf Time (s)"   : opt["inf_time"],
        "_model"         : best_model,
        "_y_proba"       : opt["y_proba"],
        "_y_pred"        : opt["y_pred"],
    }
