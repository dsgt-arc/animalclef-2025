import numpy as np
import pandas as pd
import pandas.api.types
from typing import List, Union


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def BAKS(
    y_true: List,
    y_pred: List,
    identity_test_only: List,
) -> float:
    """Computes BAKS (balanced accuracy on known samples).

    It ignores `identity_test_only` because they are unknown identities.

    Args:
        y_true (List): List of true labels.
        y_score (List): List of scores.
        identity_test_only (List): List of new identities (only in the testing set).

    Returns:
        Computed BAKS.
    """

    # Need to keep the object type due to mixed arrays
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)
    identity_test_only = np.array(identity_test_only, dtype=object)

    # Remove data in identity_test_only
    idx = np.where(~np.isin(y_true, identity_test_only))[0]
    y_true_idx = y_true[idx]
    y_pred_idx = y_pred[idx]
    if len(y_true_idx) == 0:
        return np.nan

    df = pd.DataFrame({"y_true": y_true_idx, "y_pred": y_pred_idx})

    # Compute the balanced accuracy
    accuracy = 0
    for _, df_identity in df.groupby("y_true"):
        accuracy += (
            1
            / df["y_true"].nunique()
            * np.mean(df_identity["y_pred"] == df_identity["y_true"])
        )
    return accuracy


def BAUS(
    y_true: List, y_pred: List, identity_test_only: List, new_class: Union[int, str]
) -> float:
    """Computes BAUS (balanced accuracy on unknown samples).

    It handles only `identity_test_only` because they are unknown identities.

    Args:
        y_true (List): List of true labels.
        y_score (List): List of scores.
        identity_test_only (List): List of new identities (only in the testing set).
        new_class (Union[int, str]): Name of the new class.

    Returns:
        Computed BAUS.
    """

    # Need to keep the object type due to mixed arrays
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)
    identity_test_only = np.array(identity_test_only, dtype=object)

    # Remove data not in identity_test_only
    idx = np.where(np.isin(y_true, identity_test_only))[0]
    y_true_idx = y_true[idx]
    y_pred_idx = y_pred[idx]
    if len(y_true_idx) == 0:
        return np.nan

    df = pd.DataFrame({"y_true": y_true_idx, "y_pred": y_pred_idx})

    # Compute the balanced accuracy
    accuracy = 0
    for _, df_identity in df.groupby("y_true"):
        accuracy += (
            1 / df["y_true"].nunique() * np.mean(df_identity["y_pred"] == new_class)
        )
    return accuracy


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> float:
    """Computes the geometric mean of balanced accuracies on known and unknown samples.

    Args:
        solution (pd.DataFrame): Solution dataframe.
        submission (pd.DataFrame): Submission dataframe.
        row_id_column_name (str): Column name to match the dataframes.

    Returns:
        The computed accuracy.
    """

    submission = submission.reset_index(drop=True)
    solution = solution.reset_index(drop=True)

    # Checks for correct input format
    if "image_id" not in submission.columns:
        raise ParticipantVisibleError("The submission must have column image_id.")
    if "identity" not in submission.columns:
        raise ParticipantVisibleError("The submission must have column identity.")
    if len(submission) != len(solution):
        raise ParticipantVisibleError(f"The submission length must be {len(solution)}.")
    if not np.array_equal(submission["image_id"], solution["image_id"]):
        raise ParticipantVisibleError(
            "Submission column image_id is wrong. Verify that it the same order as in sample_submission.csv."
        )
    if not pandas.api.types.is_string_dtype(submission["identity"]):
        raise ParticipantVisibleError("Submission column identity must be a string.")
    if (
        not solution["identity"]
        .apply(
            lambda x: x.startswith("SeaTurtleID2022_")
            or x.startswith("SalamanderID2025_")
            or x.startswith("LynxID2025_")
            or x == "new_individual"
        )
        .all()
    ):
        raise ParticipantVisibleError(
            "Submission column identity must start with LynxID2025_, SalamanderID2025_, SeaTurtleID2022_ or be equal to new_individual."
        )

    # Extract the data
    results = {}
    unknown_identities = solution[solution["new_identity"]]["identity"].unique()
    for name, solution_dataset in solution.groupby("dataset"):
        predictions = submission.loc[solution_dataset.index, "identity"].to_numpy()
        labels = solution_dataset["identity"].to_numpy()

        # Compute the balances accuracies on known and unknown samples
        acc_known = BAKS(labels, predictions, unknown_identities)
        acc_unknown = BAUS(labels, predictions, unknown_identities, "new_individual")

        # Fix possible nans
        if np.isnan(acc_known):
            acc_known = 0
        if np.isnan(acc_unknown):
            acc_unknown = 0

        # Save to dataframe
        results[name] = {
            "BAKS": acc_known,
            "BAUS": acc_unknown,
            "normalized": np.sqrt(acc_known * acc_unknown),
        }

    # Average the geometric mean of both accuracies
    results = pd.DataFrame(results).T
    return results["normalized"].mean()
