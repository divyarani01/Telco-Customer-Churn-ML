"""train_model() and tune_model() — exact logic from notebook cell 57."""
from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from config.config import CV_FOLDS, RANDOM_STATE
from utils.logger import get_logger

logger = get_logger(__name__)


def train_model(model, X_tr, y_tr):
    """Fit *model* on (X_tr, y_tr) and return (fitted_model, train_time_s)."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    elapsed = round(time.perf_counter() - t0, 4)
    logger.debug("train_model: fitted in %.4fs", elapsed)
    return model, elapsed


def tune_model(
    estimator,
    param_dist: dict,
    X_tr,
    y_tr,
    n_iter: int = 30,
    cv: int = CV_FOLDS,
    scoring: str = "roc_auc",
    refine_grid: dict | None = None,
):
    """Phase 1: RandomizedSearchCV. Phase 2: optional GridSearchCV refinement.

    Mirrors notebook cell 57 exactly.
    Returns the best estimator found.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    logger.info("RandomizedSearchCV (n_iter=%d, cv=%d, scoring=%s)", n_iter, cv, scoring)
    rscv = RandomizedSearchCV(
        estimator, param_dist,
        n_iter=n_iter, cv=skf,
        scoring=scoring, n_jobs=-1,
        random_state=RANDOM_STATE, verbose=0,
    )
    rscv.fit(X_tr, y_tr)
    logger.info("  RSCV best %s: %.4f | params: %s", scoring, rscv.best_score_, rscv.best_params_)
    best = rscv.best_estimator_

    if refine_grid:
        logger.info("GridSearchCV refinement ...")
        gscv = GridSearchCV(
            estimator, refine_grid, cv=skf,
            scoring=scoring, n_jobs=-1, verbose=0,
        )
        gscv.fit(X_tr, y_tr)
        if gscv.best_score_ > rscv.best_score_:
            logger.info("  GSCV improved: %.4f -> %.4f", rscv.best_score_, gscv.best_score_)
            best = gscv.best_estimator_

    return best
