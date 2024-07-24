"""Optimize hyperparameters of model."""

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np
import optuna
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn import set_config
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path, help="Path to CSV file containing data"
    )
    parser.add_argument(
        "--journal-storage-path",
        type=Path,
        help="Path to use for study journal storage",
    )
    parser.add_argument(
        "--study-name",
        default="hyperparameter-optimization-study",
        help="Name for Optuna study",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Maximum number of trials to use in optimization on each process",
    )
    parser.add_argument(
        "--n-cv-folds",
        type=int,
        default=10,
        help="Number of folds to use in computing cross validation score",
    )
    parser.add_argument(
        "--timeout-hours",
        type=int,
        default=12,
        help="Time to stop optimization study after in hours",
    )
    return parser.parse_args()


def load_training_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    subset_df = pd.read_csv(data_path)
    X_train = subset_df.drop(columns=["X_mi_m", "event_cmp", "tstop"])
    y = subset_df[["event_cmp", "tstop"]].to_numpy()
    aux = [(e1, e2) for e1, e2 in y]
    y_train = np.array(aux, dtype=[("Status", "?"), ("Survival_in_days", "<f8")])
    return X_train, y_train


def objective(trial, X_train, y_train, n_cv_folds):
    learning_rate = trial.suggest_float("learning_rate", 0.1, 1.0)
    n_estimators = trial.suggest_int("n_estimators", 2, 100)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 100)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 100)
    max_depth = trial.suggest_int("max_depth", 1, 100)
    max_features = trial.suggest_int("max_features", 1, 100)
    dropout_rate = trial.suggest_float("dropout_rate", 0, 1.0)
    subsample = trial.suggest_float("subsample", 0, 1.0)

    boost_tree = GradientBoostingSurvivalAnalysis(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features=max_features,
        dropout_rate=dropout_rate,
        subsample=subsample,
    )

    # PCA not appropriate as data variables have a clear clinical meaning
    rf = Pipeline([("boost_tree", boost_tree)])

    rf.fit(X_train, y_train)

    scores = cross_val_score(
        estimator=rf, X=X_train, y=y_train, cv=n_cv_folds, n_jobs=1
    )

    return np.mean(scores)


if __name__ == "__main__":
    set_config(display="text")  # displays text representation of estimators
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    args = parse_arguments()
    X_train, y_train = load_training_data(args.data_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(args.journal_storage_path),
    )
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="maximize"
    )
    study.optimize(
        partial(
            objective, X_train=X_train, y_train=y_train, n_cv_folds=args.n_cv_folds
        ),
        n_trials=args.n_trials,
        timeout=args.timeout_hours * 3600,
        gc_after_trial=True,
        n_jobs=1,
    )
