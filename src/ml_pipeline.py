import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
#from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
from joblib import parallel_backend


pipe = Pipeline([
    ("pre", StandardScaler()),
    ("clf", LogisticRegression())  # placeholder; will be overridden by param grids
])

clf_name_to_init = {
    'LogisticRegression': LogisticRegression(
        max_iter=5000,
        n_jobs=-1,
        random_state=614,
        tol=1e-3
    ),
    'GaussianNB': GaussianNB(),
    'ComplementNB': ComplementNB(),
    'BernoulliNB': BernoulliNB(),
    'RandomForestClassifier': RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=614
    ),
    'XGBClassifier': XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=614,
    ),
    #'LGBMClassifier': LGBMClassifier(
    #    n_jobs=-1,
    #    verbose=0
    #),
    #'CatBoostClassifier': CatBoostClassifier(
    #        verbose=0,
    #        loss_function="Logloss",
    #        thread_count=-1
    #    )
}

scorers = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "neg_log_loss": "neg_log_loss",
}
for max_pct in [10, 25, 50]:
    scorers[f"roc_auc_{max_pct}"] = make_scorer(
        roc_auc_score,
        response_method=("decision_function", "predict_proba"),
        max_fpr=max_pct/100
    )

def get_initialized_random_search(pipe, param_space, draw, scorers, verbosity=1):
    search = RandomizedSearchCV(
        random_state=614,
        estimator=pipe,
        param_distributions=param_space,
        n_iter=draw,
        scoring=scorers,
        refit="pr_auc",
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=614),
        n_jobs=-1,
        verbose=verbosity,
        error_score='raise',
        return_train_score=False,
        pre_dispatch='n_jobs',  # don’t pre-spawn a big task queue
    )
    return search

def run_random_search(pipe, feature_df, y, param_spaces, draws, scorers, verbosity=1):
    X_np = np.asarray(feature_df, dtype=np.float64, order='C')
    y_np = np.asarray(y)

    grid_df = pd.DataFrame()
    for param_space, draw in zip(
        param_spaces,
        draws
    ):
        search = get_initialized_random_search(
            pipe=pipe,
            param_space=param_space,
            draw=draw,
            scorers=scorers,
            verbosity=verbosity)
        with parallel_backend("threading"):   # <-- key change
            search.fit(X_np, y_np)  # Keep n_jobs=-1 in RandomizedSearchCV
        grid_df = pd.concat([grid_df, pd.DataFrame(search.cv_results_)])
    grid_df['n_features'] = len(feature_df.columns)

    return grid_df
