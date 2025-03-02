import typer
from typing_extensions import Annotated
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .transform import WrappedDino
from animalclef.spark import get_spark


def transform(model, df, features) -> DataFrame:
    transformed = model.transform(df)

    for c in features:
        # check if the feature is a vector and convert it to an array
        if "array" in transformed.schema[c].simpleString():
            continue
        transformed = transformed.withColumn(c, vector_to_array(F.col(c)))
    return transformed


def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    sample_id: Annotated[int, typer.Option(help="Sample ID")] = None,
    num_sample_ids: Annotated[int, typer.Option(help="Number of sample IDs")] = 50,
    num_partitions: Annotated[int, typer.Option(help="Number of partitions")] = 20,
):
    df = get_spark().read.parquet(input_path)

    # we'll write out a partitioned dataset
    if sample_id is not None:
        df = (
            df.withColumn(
                "sample_id",
                F.crc32(F.col("image_id").cast("string")) % num_sample_ids,
            )
            .where(F.col("sample_id") == sample_id)
            .drop("sample_id")
        )
        output_path = f"{output_path}/sample_id={sample_id}"

    # transform the dataframe and write to disk
    transformed = transform(
        Pipeline(
            stages=[
                WrappedDino(
                    input_col="data",
                    output_col="cls_token",
                    batch_size=batch_size,
                ),
            ]
        ),
        df,
        "data",
    ).select("image_id", "cls_token")

    transformed.printSchema()
    transformed.explain()
    (
        transformed.repartition(num_partitions)
        .write.mode("overwrite")
        .parquet(output_path)
    )
