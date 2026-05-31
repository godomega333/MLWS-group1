"""Shared path constants and Kaggle-submission helper for AI3023 MLW Group 1
supplementary experiments.

Each experiment script writes its predictions to a CSV under RESULTS_DIR.
LB scoring is obtained by submitting the CSV to the Kaggle Spaceship Titanic
competition page (https://www.kaggle.com/competitions/spaceship-titanic).
"""
from __future__ import annotations

import os

DATA_DIR = "spaceship-titanic-dataset"
RESULTS_DIR = os.path.join("experiments", "results")
FIGURES_DIR = os.path.join("experiments", "figures")


def kaggle_submission_notice(pred_csv, label=None):
    """Print a notice telling the operator to submit pred_csv to Kaggle."""
    tag = f" (label={label})" if label else ""
    print(f"[Kaggle submission]{tag} Saved {pred_csv}.")
    print("  Submit this CSV to https://www.kaggle.com/competitions/spaceship-titanic")
    print("  to obtain the public-leaderboard accuracy.")
