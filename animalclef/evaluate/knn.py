import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F
from tqdm import tqdm
from typer import Typer

from animalclef.dataset import split_reid_data, summarize_split
from animalclef.metrics import BAKS, BAUS
from animalclef.spark import get_spark

app = Typer()


def get_avg_distance_to_neighbor(train_df, num_neighbors=1):
    @lru_cache(maxsize=16)
    def get_index(individual):
        """get all embeddings that do not belong to an individual. This will form
        the minimum intercluster distance."""
        sub = train_df[train_df.identity != individual]
        # Use the dimension of the embedding vectors
        embedding_dim = len(sub.embeddings.iloc[0])
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.stack(sub.embeddings.values))
        return index

    distances = []
    # order the training set by identity so that our lru cache is more efficient
    train_df = train_df.sort_values(by="identity")
    for row in tqdm(train_df.itertuples(), total=len(train_df)):
        # get the index for the individual
        index = get_index(row.identity)
        # search for the nearest neighbor in the training set
        dist, _ = index.search(np.array([row.embeddings]), num_neighbors)
        # get the average distance to the nearest neighbors
        distances.append(np.median(dist))
    return np.array(distances)


def summarize_distances(distances):
    q1 = np.percentile(distances, 25)
    q2 = np.percentile(distances, 50)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    mad = np.median(np.abs(distances - np.median(distances)))

    return {
        "min_distance": float(distances.min()),
        "max_distance": float(distances.max()),
        "mean_distance": float(distances.mean()),
        "std_distance": float(distances.std()),
        "iqr": float(iqr),
        "q1": float(q1),
        "q3": float(q3),
        "median": float(q2),
        "mad": float(mad),
    }


def make_prediction(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    median: float,
    mad: float,
    threshold: float,
    num_neighbors: int = 1,
    new_label: str = "new_individual",
) -> np.ndarray:
    X_train = np.stack(train_df.embeddings.values)
    X_other = np.stack(other_df.embeddings.values)

    index = faiss.IndexFlatL2(X_train.shape[1])
    index.add(X_train)

    dist, idx = index.search(X_other, num_neighbors)

    predictions = []
    for i, (dists, ids) in enumerate(zip(dist, idx)):
        # filter ids where the distance is above the threshold
        filtered_ids = []
        for dist, id_ in zip(dists, ids):
            score = (dist - median) / (mad * 1.4826)
            if score < threshold:
                filtered_ids.append(id_)

        if len(filtered_ids) == 0:
            predictions.append(new_label)
        else:
            # get the most common identity among the k nearest neighbors
            counts = Counter(train_df.iloc[filtered_ids]["identity"].values)
            predictions.append(counts.most_common(1)[0][0])
    return np.array(predictions)


def search_threshold(train_df, other_df, median, mad, thresholds, num_neighbors=1):
    identity_other_only = sorted(
        set(other_df.identity.unique()) - set(train_df.identity.unique())
    )

    scores = []
    for threshold in tqdm(thresholds):
        predictions = make_prediction(
            train_df, other_df, median, mad, threshold, num_neighbors=num_neighbors
        )
        baks = BAKS(other_df["identity"].values, predictions, identity_other_only)
        baus = BAUS(
            other_df["identity"].values,
            predictions,
            identity_other_only,
            "new_individual",
        )
        scores.append(
            {
                "threshold": threshold,
                "baks": baks,
                "baus": baus,
                "crossover_score": 1 - abs(baks - baus),
                # geometric average
                "score": np.sqrt(baks * baus),
            }
        )
    return pd.DataFrame(scores)


def plot_threshold_score(df, col="threshold"):
    best_score_row = df.iloc[df["score"].idxmax()]
    best_score_threshold = best_score_row[col]

    best_crossover_score_row = df.iloc[df["crossover_score"].idxmax()]
    best_crossover_score_threshold = best_crossover_score_row[col]

    # plot baus and baks
    plt.plot(
        df[col],
        df["baks"],
        label=f"BAKS (best {df['baks'].max():.2f})",
    )
    plt.plot(
        df[col],
        df["baus"],
        label=f"BAUS (best {df['baus'].max():.2f})",
    )
    plt.plot(
        df[col],
        df["score"],
        label=f"score (best {df["score"].max():.2f})",
    )
    # line at the best threshold
    plt.axvline(
        best_score_threshold,
        color="red",
        linestyle="--",
        label=f"best threshold: {best_score_threshold:.2f}",
    )
    plt.axvline(
        best_crossover_score_threshold,
        color="blue",
        linestyle="--",
        label=f"crossover threshold: {best_crossover_score_threshold:.2f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Geometric mean of BAKS and BAUS")
    plt.title("Threshold vs Geometric mean of BAKS and BAUS")
    plt.legend()


def experiment_threshold(train_df, val_df, num_neighbors=1):
    # get stats on the distances
    distances = get_avg_distance_to_neighbor(train_df, num_neighbors=num_neighbors)
    median = np.median(distances)
    mad = np.median(np.abs(distances - median))
    thresholds = np.linspace(-2, 8, 100)
    mad_threshold_df = search_threshold(
        train_df,
        val_df,
        median=median,
        mad=mad,
        thresholds=thresholds,
        num_neighbors=num_neighbors,
    )
    mad_threshold_df["num_neighbors"] = num_neighbors
    mad_threshold_df["median"] = median
    mad_threshold_df["mad"] = mad
    return mad_threshold_df, distances


@app.command()
def run_prediction(
    metadata_path: str, embedding_path: str, output_path: str, num_neighbors: int
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    spark = get_spark(cores=2, memory="1g")
    pdf = (
        spark.read.csv(metadata_path, header=True, inferSchema=True)
        .join(
            spark.read.parquet(embedding_path).select("image_id", "embeddings"),
            on="image_id",
            how="inner",
        )
        .select(
            "image_id",
            "identity",
            "dataset",
            "embeddings",
            F.count("image_id")
            .over(Window.partitionBy("identity"))
            .alias("identity_count"),
        )
        .toPandas()
    )

    # do the test split
    cond = (~pdf.identity.isnull()) & (pdf.identity_count > 2)
    train_df, val_df, test_df = split_reid_data(pdf[cond])
    reid_split_summary = summarize_split(train_df, val_df, test_df)
    reid_split_summary.to_csv(f"{output_path}/reid_split_summary.csv", index=False)

    # validation thresholds
    thresholds_df, distances = experiment_threshold(
        train_df, val_df, num_neighbors=num_neighbors
    )
    thresholds_df.to_csv(f"{output_path}/val_thresholds.csv")
    (output_path / "val_distance_summary.json").write_text(
        json.dumps(summarize_distances(distances), indent=2)
    )
    plot_threshold_score(thresholds_df, col="threshold")
    plt.savefig(f"{output_path}/val_thresholds.png")
    plt.close()
    row = thresholds_df.iloc[thresholds_df["score"].idxmax()]
    # write the best row to a json file
    (output_path / "val_best_threshold.json").write_text(
        json.dumps(row.to_dict(), indent=2)
    )

    # test thresholds
    thresholds_df, distances = experiment_threshold(
        pd.concat([train_df, val_df]), test_df, num_neighbors=num_neighbors
    )
    thresholds_df.to_csv(f"{output_path}/test_thresholds.csv")
    (output_path / "test_distance_summary.json").write_text(
        json.dumps(summarize_distances(distances), indent=2)
    )
    plot_threshold_score(thresholds_df, col="threshold")
    plt.savefig(f"{output_path}/test_thresholds.png")
    plt.close()

    # make the prediction
    row = thresholds_df.iloc[thresholds_df["score"].idxmax()]
    # write the best row to a json file
    (output_path / "test_best_threshold.json").write_text(
        json.dumps(row.to_dict(), indent=2)
    )
    known_df = pdf[pdf.identity.notnull()]
    unknown_df = pdf[pdf.identity.isnull()]
    predictions = make_prediction(
        known_df,
        unknown_df,
        row["median"],
        row["mad"],
        row["threshold"],
        num_neighbors=num_neighbors,
    )
    # count how many are predicted as "unknown"
    known_counts = {
        "unknown": len(predictions[predictions == "new_individual"]),
        "known": len(predictions[predictions != "new_individual"]),
        "total": len(predictions),
    }
    (output_path / "test_known_counts.json").write_text(
        json.dumps(known_counts, indent=2)
    )
    unknown_df["identity"] = predictions
    unknown_df[["image_id", "identity"]].to_csv(
        f"{output_path}/prediction.csv", index=False, header=True
    )

    # write _SUCCESS
    (output_path / "_SUCCESS").touch()


if __name__ == "__main__":
    app()
