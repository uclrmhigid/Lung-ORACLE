"""Optimize hyperparameters of model."""

import argparse
import logging
import sys
import time
from functools import partial
from pathlib import Path

import optuna


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--timeout-minutes",
        type=int,
        default=5,
        help="Time to stop optimization study after in minutes",
    )
    return parser.parse_args()


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    time.sleep(1)
    return (x - 2) ** 2

if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    args = parse_arguments()
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(str(args.journal_storage_path)),
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout_minutes * 60,
        gc_after_trial=True,
        n_jobs=1,
    )
