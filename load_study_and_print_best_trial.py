"""Load Optuna study from journal file storage and print best trial details."""

import argparse
from pathlib import Path
import optuna


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal-storage-path",
        type=Path,
        help="Path to study journal storage file",
    )
    parser.add_argument(
        "--study-name",
        default="hyperparameter-optimization-study",
        help="Name for Optuna study",
    )
    return parser.parse_args()


def load_study_and_print_best_trial(journal_storage_path, study_name):
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(str(journal_storage_path)),
    )
    study = optuna.study.load_study(study_name=study_name, storage=storage)
    print("# Best trial")
    print(f"  Value: {study.best_trial.value:.5f}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    args = parse_arguments()
    load_study_and_print_best_trial(args.journal_storage_path, args.study_name)
