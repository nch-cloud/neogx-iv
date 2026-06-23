"""
random_grid_search_cli.py

Minimal-diff conversion of your notebook into a reusable Python module/CLI.
- Preserves your exact feature loading, cohort/ID filtering, param spaces, scorers, and threading backend.
- Lets you import run_all_searches(...) from another orchestration notebook, or run as a script.

Examples
--------
As a script:
    python random_grid_search_cli.py \
        --data-root ../data \
        --models lr,gnb,rf,xgb \
        --draws 30,10,60,120 \
        --target label__dx_or_lab_or_consult \
        --save-dir ../data/grid-search-results

From a notebook:
    from random_grid_search_cli import run_all_searches
    grid_df, best_model, best_params, test_metrics = run_all_searches()

Notes
-----
- Requires `ml_pipeline` module that exports `pipe` and `clf_name_to_init`.
- XGBoost is optional; if not installed and requested, it will be skipped with a warning.
- Default settings mirror your notebook.
"""
from __future__ import annotations

import os
import re
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Prevent CPU oversubscription when using n_jobs>1
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from joblib import parallel_backend
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer, average_precision_score

# Your pipeline bits (expected to be available)
from ml_pipeline import pipe, clf_name_to_init  # type: ignore

# ------------------------------ Config ------------------------------ #

@dataclass
class FeatureConfig:
    vocab: str = "phecode"
    representatives: str = "depth2"
    encoding: str = "depth"
    dx_type: str = "Encounter Diagnosis"  # or "Hospital Problem"
    level4_day: int = 1
    binary: bool = True


def _discover_feature_file(feature_dir: Path, cfg: FeatureConfig) -> str:
    files = os.listdir(feature_dir)
    for attr in ["representatives", "encoding", "dx_type", "level4_day"]:
        val = getattr(cfg, attr)
        files = [f for f in files if f"{attr}={val}" in f]
    if not files:
        raise FileNotFoundError("No feature matrix found matching the provided FeatureConfig.")
    elif len(files) > 1:
        for file in files:
            print(file)
        raise BaseException("multiple files available")
    else:
        return files[0]


# --------------------------- Data Loading --------------------------- #

def load_features_and_labels(
    data_root: Path,
    cfg: FeatureConfig,
    target_var_name: str,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    feature_dir = data_root / "feature_matrices"
    if cfg.vocab != "phecode":
        raise NotImplementedError("Only 'phecode' vocab flow was present in the notebook.")

    feature_fname = _discover_feature_file(feature_dir, cfg)

    static_feature_df = pd.read_csv(feature_dir / "static_features.csv.gz", index_col="pat_id")
    pheno_feature_df = pd.read_csv(feature_dir / feature_fname, index_col="pat_id")
    if cfg.binary:
        pheno_feature_df = (pheno_feature_df > 0).astype(int)

    feature_df = pd.concat([static_feature_df, pheno_feature_df], axis=1)

    # Drop specific features if present (mirrors your cell)
    for feat in ["gestational_age_complete_weeks"]:
        if feat in feature_df.columns:
            del feature_df[feat]

    # Remove rows with any nulls
    null_mask = feature_df.isna().any(axis=1)
    print("rows with nulls:", int(null_mask.sum()))
    feature_df = feature_df.loc[~null_mask].copy()

    # Target
    outcome_df = pd.read_csv(data_root / "outcome-dataframe.csv", index_col="pat_id")
    y = outcome_df[target_var_name]

    return feature_df, y, feature_fname


def train_test_ids(feature_df: pd.DataFrame, data_root: Path) -> Tuple[pd.Index, pd.Index, pd.Index]:
    cohort_df = pd.read_csv(data_root / "cohort-dataframe.csv", index_col="pat_id")
    outcome_df = pd.read_csv(data_root / "outcome-dataframe.csv", index_col="pat_id")

    ids_valid = feature_df.index
    ids_trisomy = outcome_df.query("trisomy_binary == 1").index

    ids_tr = (
        cohort_df.query("birth_year < 2022").index
        .difference(ids_trisomy)
        .intersection(ids_valid)
    )
    ids_te = (
        cohort_df.query("birth_year.isin([2022, 2023])").index
        .difference(ids_trisomy)
        .intersection(ids_valid)
    )
    return ids_tr, ids_te, ids_valid


# --------------------------- Param Spaces --------------------------- #

def build_param_spaces(y_train: pd.Series) -> Dict[str, Dict]:
    # Logistic Regression
    space_lr = {
        "clf": [clf_name_to_init["LogisticRegression"]],
        "clf__class_weight": [None, "balanced"],
        "clf__solver": ["saga"],
        "clf__penalty": ["l1", "l2", None],
        "clf__C": loguniform(1e-3, 1e2),
        "clf__fit_intercept": [True, False],
    }

    # Naive Bayes
    space_gnb = {
        "clf": [clf_name_to_init["GaussianNB"]],
        "clf__var_smoothing": loguniform(1e-12, 1e-6),
    }
    space_cnb = {
        "clf": [clf_name_to_init.get("ComplementNB")],
        "clf__alpha": loguniform(1e-3, 10),
        "clf__norm": [True, False],
    }
    space_bnb = {
        "clf": [clf_name_to_init.get("BernoulliNB")],
        "clf__alpha": loguniform(1e-3, 10),
        "clf__binarize": [None, 0.0, 0.01, 0.1],
    }

    # Random Forest
    space_rf = {
        "clf": [clf_name_to_init["RandomForestClassifier"]],
        "clf__n_estimators": randint(200, 1001),
        "clf__max_depth": [None] + list(randint(5, 51).rvs(10)),
        "clf__min_samples_split": randint(2, 21),
        "clf__min_samples_leaf": randint(1, 11),
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
        "clf__bootstrap": [True],
        "clf__class_weight": [None, "balanced_subsample"],
    }

    # XGBoost (optional)
    try:
        import xgboost  # noqa: F401
        neg, pos = np.bincount(y_train.astype(int))
        spw = float(max(1.0, neg / max(1, pos)))
        space_xgb = {
            "clf": [clf_name_to_init["XGBClassifier"]],
            "clf__n_estimators": randint(300, 1001),
            "clf__learning_rate": loguniform(1e-2, 3e-1),
            "clf__max_depth": randint(3, 10),
            "clf__min_child_weight": randint(1, 11),
            "clf__subsample": uniform(0.5, 0.5),
            "clf__colsample_bytree": uniform(0.5, 0.5),
            "clf__gamma": [0.0, 0.1, 0.3, 1.0],
            "clf__reg_alpha": loguniform(1e-4, 10),
            "clf__reg_lambda": loguniform(1e-1, 10),
            "clf__scale_pos_weight": [1, spw],
        }
    except Exception:
        space_xgb = None

    spaces = {
        "lr": space_lr,
        "gnb": space_gnb,
        "cnb": space_cnb,
        "bnb": space_bnb,
        "rf": space_rf,
    }
    if space_xgb is not None:
        spaces["xgb"] = space_xgb
    return spaces


# ----------------------------- Scorers ------------------------------ #

def build_scorers() -> Dict[str, object]:
    scorers = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
    }
    for max_pct in [10, 25, 50]:
        scorers[f"roc_auc_{max_pct}"] = make_scorer(
            roc_auc_score,
            response_method=("decision_function", "predict_proba"),
            max_fpr=max_pct / 100,
        )
    return scorers


# --------------------------- Search Runner -------------------------- #

def run_all_searches(
    data_root: str = "../data",
    target_var_name: str = "label__dx_or_lab_or_consult",
    feature_cfg: FeatureConfig = FeatureConfig(),
    models: List[str] = None,
    draws: List[int] = None,
    random_state: int = 614,
    refit_metric: str = "pr_auc",
    cv_splits: int = 3,
    n_jobs: int = -1,
    verbose: int = 10,
    pre_dispatch: str = "n_jobs",
) -> Tuple[pd.DataFrame, object, dict, Dict[str, float]]:
    """Runs the randomized searches and returns (grid_df, best_model, best_params, test_metrics)."""
    data_root = Path(data_root)

    # Load data
    feature_df, y, feature_fname = load_features_and_labels(data_root, feature_cfg, target_var_name)
    ids_tr, ids_te, _ = train_test_ids(feature_df, data_root)

    X_np = np.asarray(feature_df.loc[ids_tr], dtype=np.float64, order="C")
    y_np = np.asarray(y.loc[ids_tr])

    # Param spaces based on the *training* target for imbalance hints
    spaces = build_param_spaces(pd.Series(y_np))

    # Which models to run & how many draws each
    if models is None:
        models = ["lr", "gnb", "rf", "xgb"]
    if draws is None:
        draws = [30, 10, 60, 120][: len(models)]
    if len(models) != len(draws):
        raise ValueError("`models` and `draws` must be the same length.")

    # Build scorers
    scorers = build_scorers()

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    grid_df = pd.DataFrame()
    best_search = None

    for model_key, n_iter in zip(models, draws):
        space = spaces.get(model_key)
        if space is None:
            print(f"[warn] Skipping '{model_key}' (space not available, maybe xgboost isn't installed?)")
            continue
        search = RandomizedSearchCV(
            random_state=random_state,
            estimator=pipe,
            param_distributions=space,
            n_iter=n_iter,
            scoring=scorers,
            refit=refit_metric,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            error_score="raise",
            return_train_score=False,
            pre_dispatch=pre_dispatch,
        )
        # Match your notebook's parallelization behavior
        with parallel_backend("threading"):
            search.fit(X_np, y_np)

        # Accumulate results
        grid_df = pd.concat([grid_df, pd.DataFrame(search.cv_results_)], ignore_index=True)
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search = search

    if best_search is None:
        raise RuntimeError("No searches executed successfully.")

    # Sort and assemble outputs like your notebook
    score_cols = [c for c in grid_df.columns if "mean_test" in c]
    param_cols = [c for c in grid_df.columns if c.startswith("param_")]
    if "mean_test_pr_auc" in grid_df.columns:
        grid_df = grid_df.sort_values("mean_test_pr_auc", ascending=False)
    grid_df = grid_df.reset_index(drop=True)

    # Fit final model on full train with best params
    params = grid_df.iloc[0]["params"]
    model = pipe.set_params(**params)
    model.fit(feature_df.loc[ids_tr], y.loc[ids_tr])

    # Evaluate on test
    p_te = model.predict_proba(feature_df.loc[ids_te])[:, 1]
    test_metrics = {
        "roc_auc": float(roc_auc_score(y_true=y.loc[ids_te], y_score=p_te)),
        "pr_auc": float(average_precision_score(y_true=y.loc[ids_te], y_score=p_te)),
    }

    return grid_df, model, params, test_metrics


# ------------------------------ Saving ------------------------------ #

def default_save_prefix(feature_fname: str, feature_cfg: FeatureConfig, target_var_name: str) -> str:
    save_config: Dict[str, str] = {}
    save_config.update(dict(re.findall(r"([\w\_]+)\=([\w\s]+)", feature_fname)))
    if feature_cfg.binary:
        save_config["encoding"] = "binary"
    save_config["target_variable"] = target_var_name
    prefix = "grid_search_results-" + "-".join([f"{k}={v}" for k, v in save_config.items()])
    return prefix


def save_grid_results(grid_df: pd.DataFrame, out_dir: Path, prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}.csv.gz"
    grid_df.to_csv(path, compression="gzip", index=False)
    return path


# ------------------------------- CLI -------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run randomized searches (minimal-diff from notebook).")
    p.add_argument("--data-root", type=str, default="../data", help="Root folder containing feature_matrices/, cohort/outcome CSVs")
    p.add_argument("--target", type=str, default="label__dx_or_lab_or_consult", help="Target variable/column name")
    p.add_argument("--models", type=str, default="lr,gnb,rf,xgb", help="Comma list from {lr,gnb,cnb,bnb,rf,xgb}")
    p.add_argument("--draws", type=str, default="30,10,60,120", help="Comma list of n_iter per model, same length as --models")
    p.add_argument("--save-dir", type=str, default="../data/grid-search-results", help="Where to write the grid results CSV")
    p.add_argument("--no-save", action="store_true", help="Do not write results to disk")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    draws = [int(x) for x in args.draws.split(",") if x.strip()]

    # Run searches
    grid_df, model, best_params, test_metrics = run_all_searches(
        data_root=args.data_root,
        target_var_name=args.target,
        models=models,
        draws=draws,
    )

    # Recreate save prefix using the discovered feature filename and config for transparency
    # (We need to recompute the feature filename here; easiest is to call the same discovery)
    feature_df, _, feature_fname = load_features_and_labels(Path(args.data_root), FeatureConfig(), args.target)
    prefix = default_save_prefix(feature_fname, FeatureConfig(), args.target)

    if not args.no_save:
        out_path = save_grid_results(grid_df, Path(args.save_dir), prefix)
        print(f"Saved grid results -> {out_path}")

    print("Best params:", best_params)
    print("Test ROC AUC:", test_metrics["roc_auc"])
    print("Test PR AUC:", test_metrics["pr_auc"])


if __name__ == "__main__":
    main()
