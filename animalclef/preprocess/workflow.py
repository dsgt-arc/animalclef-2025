import typer
from typing_extensions import Annotated
from pyspark.sql import functions as F
from pathlib import Path
from animalclef.spark import get_spark

app = typer.Typer(name="preprocess", no_args_is_help=True)


@app.command("parquet")
def main(
    input_path: Annotated[str, typer.Argument(help="Input root directory")],
    output_path: Annotated[str, typer.Argument(help="Output root directory")],
    num_partitions: Annotated[int, typer.Option(help="Number of partitions")] = 20,
):
    spark = get_spark()
    metadata = spark.read.csv(
        f"{input_path}/metadata.csv",
        header=True,
        inferSchema=True,
    )
    metadata.printSchema()
    """
        root
        |-- image_id: string (nullable = true)
        |-- identity: string (nullable = true)
        |-- path: string (nullable = true)
        |-- date: string (nullable = true)
        |-- orientation: string (nullable = true)
        |-- species: string (nullable = true)
        |-- split: string (nullable = true)
        |-- dataset: string (nullable = true)
    """
    metadata_output_path = f"{output_path}/parquet/metadata"
    metadata.repartition(4).write.parquet(metadata_output_path, mode="overwrite")

    @F.udf("string")
    def relative_path(path):
        return (
            Path(path.split("file:")[1])
            .relative_to(Path(input_path).absolute())
            .as_posix()
        )

    images = (
        spark.read.format("binaryFile")
        # image/{dataset}/{split}/{image}
        .option("pathGlobFilter", "*")
        .option("recursiveFileLookup", True)
        .load(f"{input_path}/images")
    )
    images.printSchema()
    """
        root
        |-- path: string (nullable = true)
        |-- modificationTime: timestamp (nullable = true)
        |-- length: long (nullable = true)
        |-- content: binary (nullable = true)
    """

    image_output_path = f"{output_path}/parquet/images"
    (
        images.withColumn("path", relative_path(F.col("path")))
        .join(
            metadata.select("image_id", "path", "dataset", "split"),
            on="path",
            how="left",
        )
        .repartition(num_partitions, "dataset", "split")
        .write.partitionBy("dataset", "split")
        .parquet(image_output_path, mode="overwrite")
    )

    # sanity check
    assert (
        spark.read.parquet(metadata_output_path).count()
        == spark.read.parquet(image_output_path).count()
    ), "metadata and images count mismatch"
