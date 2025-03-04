import typer
from typing_extensions import Annotated
from pyspark.ml import Pipeline
from pyspark.sql import functions as F

from .transform import WrappedDino
from animalclef.spark import get_spark

app = typer.Typer(name="embed", no_args_is_help=True)


# def features_to_array(df, features: list) -> DataFrame:
#     for c in features:
#         # check if the feature is a vector and convert it to an array
#         if "array" in df.schema[c].simpleString():
#             continue
#         df = df.withColumn(c, vector_to_array(F.col(c)))
#     return df


@app.command("dinov2")
def embed_dinov2(
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
    transformed = (
        Pipeline(
            stages=[
                WrappedDino(
                    input_col="content",
                    output_col="token",
                    batch_size=batch_size,
                ),
            ]
        )
        .fit(df)
        .transform(df)
        .select("image_id", "token")
    )

    transformed.printSchema()
    transformed.explain()
    (
        transformed.repartition(num_partitions)
        .write.mode("overwrite")
        .parquet(output_path)
    )
