"""
This module provides functionality to split datasets for open-set re-identification tasks.
The splitting strategy ensures:
- No data leakage by splitting based on individual IDs
- Proper evaluation of both known and unknown individuals
- Clear separation between training, validation, and test sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_reid_data(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.5,
    known_ratio: float = 0.8,
    group_col: str = "identity",
    image_col: str = "image_id",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training, validation, and test sets for open-set re-identification tasks.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the dataset.
    train_ratio (float): The ratio of the dataset to be used for training.
    val_ratio (float): The ratio of the temporary test set to be used for validation.
    known_ratio (float): The ratio of images of known individuals to be used for training.
    group_col (str): The column name representing unique identities.
    image_col (str): The column name representing image identifiers.
    seed (int): The random seed for reproducibility.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test dataframes.

    The splitting process follows these steps:
    1. Split unique individual IDs into training and temporary test sets.
    2. Further split the temporary test set into validation and final test sets for unknown individuals.
    3. Split images of known individuals into training, validation, and test sets.
    4. Combine known and unknown dataframes to create final validation and test sets.
    """
    # 1. Split into Train and (Temp) Test sets based on individuals (groups)
    unique_ids = df[group_col].unique()
    train_ids, temp_test_ids = train_test_split(
        unique_ids, train_size=train_ratio, random_state=seed
    )

    # 2. Split (Temp) Test into Validation and Final Test sets (unknown individuals).
    val_unknown_ids, test_unknown_ids = train_test_split(
        temp_test_ids, train_size=val_ratio, random_state=seed + 1
    )
    # 3. Split *images* of known individuals.
    val_known_images = []
    test_known_images = []
    train_images = []
    for indiv_id in sorted(train_ids):
        images = df[df[group_col] == indiv_id][image_col].tolist()
        if len(images) == 1:
            # If a "known ID" has only one image, it must go to the training gallery.
            # It cannot also be a query for itself if it's the only sample.
            train_images.extend(images)
        else:
            # Split images for this known ID into gallery and a pool for queries
            gallery_samples, query_samples_for_id = train_test_split(
                images, train_size=known_ratio, random_state=seed + 2
            )
            train_images.extend(gallery_samples)

            if len(query_samples_for_id) == 1:
                # If only one image is left for queries from this known ID,
                # assign it all to validation_known_images (or test, or alternate).
                val_known_images.extend(query_samples_for_id)
            elif len(query_samples_for_id) > 1:
                # If 2 or more images are left for queries, split them 50/50
                val_k_queries, test_k_queries = train_test_split(
                    query_samples_for_id, test_size=0.5, random_state=seed + 3
                )
                val_known_images.extend(val_k_queries)
                test_known_images.extend(test_k_queries)

    # Create the known and unknown dfs
    val_df_known = df[df[image_col].isin(val_known_images)]
    test_df_known = df[df[image_col].isin(test_known_images)]
    train_df = df[df[image_col].isin(train_images)]
    val_df_unknown = df[df[group_col].isin(val_unknown_ids)]
    test_df_unknown = df[df[group_col].isin(test_unknown_ids)]

    # 4. Combine to create final DataFrames
    val_df = pd.concat([val_df_known, val_df_unknown])
    test_df = pd.concat([test_df_known, test_df_unknown])

    return train_df, val_df, test_df


def summarize_split(train_df, val_df, test_df, id_col="identity", image_col="image_id"):
    """
    Generate a comprehensive summary of the dataset split, validating its integrity for
    open-set re-identification tasks.

    The summary includes:
    - Basic statistics (number of individuals and images per split)
    - Image overlap analysis to detect data leakage
    - Distribution of known vs unknown individuals in each split

    Parameters:
    train_df (pd.DataFrame): Training set dataframe
    val_df (pd.DataFrame): Validation set dataframe
    test_df (pd.DataFrame): Test set dataframe
    id_col (str): Column name for individual identifiers
    image_col (str): Column name for image identifiers

    Returns:
    pd.DataFrame: A summary table with the following metrics for each split:
        - Number of unique individuals
        - Number of images
        - Image overlap counts and percentages between splits
        - Count of known individuals (present in training)
        - Count of unknown individuals (not in training)
    """
    # Create base summary with individual and image counts
    summary = pd.DataFrame(
        {
            "Split": ["Train", "Validation", "Test"],
            "Num Individuals": [
                train_df[id_col].nunique(),
                val_df[id_col].nunique(),
                test_df[id_col].nunique(),
            ],
            "Num Images": [len(train_df), len(val_df), len(test_df)],
        }
    )

    # Extract unique identifiers and images for overlap analysis
    train_ids = set(train_df[id_col])
    val_ids = set(val_df[id_col])
    test_ids = set(test_df[id_col])

    train_images = set(train_df[image_col])
    val_images = set(val_df[image_col])
    test_images = set(test_df[image_col])

    # Calculate image overlaps between splits
    summary["Train Image Overlap"] = [
        len(train_images.intersection(train_images)),
        len(train_images.intersection(val_images)),
        len(train_images.intersection(test_images)),
    ]
    summary["Val Image Overlap"] = [
        len(val_images.intersection(train_images)),
        len(val_images.intersection(val_images)),
        len(val_images.intersection(test_images)),
    ]
    summary["Test Image Overlap"] = [
        len(test_images.intersection(train_images)),
        len(test_images.intersection(val_images)),
        len(test_images.intersection(test_images)),
    ]

    # Convert overlaps to percentages for easier interpretation
    for split, total in zip(
        ["Train", "Val", "Test"], [len(train_images), len(val_images), len(test_images)]
    ):
        summary[f"{split} Image %"] = (
            summary[f"{split} Image Overlap"] / total * 100
        ).round(2)

    # Track distribution of known vs unknown individuals
    summary["Known Individuals"] = [
        len(train_ids),
        len(val_ids.intersection(train_ids)),
        len(test_ids.intersection(train_ids)),
    ]
    summary["Unknown Individuals"] = [
        0,  # Training set has no unknown individuals by design
        len(val_ids - train_ids),
        len(test_ids - train_ids),
    ]
    return summary
