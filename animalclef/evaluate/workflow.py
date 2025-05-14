import luigi
import typer
from .embed import get_embeddings
from .knn import run_prediction
from pathlib import Path

app = typer.Typer()


class SharedParamsMixin:
    metadata_path = luigi.Parameter(
        description="Path to the metadata file containing image IDs and embeddings."
    )
    embedding_path = luigi.Parameter(
        description="Path to the file containing the embeddings."
    )
    output_path = luigi.Parameter(
        description="Path to the output file where the embeddings will be saved."
    )


class EmbeddingTask(luigi.Task, SharedParamsMixin):
    projection_head_path = luigi.Parameter(
        description="Path to the projection head model file."
    )
    batch_size = luigi.IntParameter(
        default=32,
        description="Batch size for loading the embeddings.",
    )
    embed_dim = luigi.IntParameter(
        default=128,
        description="Dimension of the embeddings.",
    )
    projector = luigi.Parameter(
        default="linear",
        description="Type of projection head to use (linear or nonlinear).",
    )

    def output(self):
        return {
            "parquet": luigi.LocalTarget(f"{self.output_path}/embeddings.parquet"),
            "png": luigi.LocalTarget(f"{self.output_path}/embeddings.png"),
        }

    def run(self):
        get_embeddings(
            metadata_path=self.metadata_path,
            embedding_path=self.embedding_path,
            projection_head_path=self.projection_head_path,
            output_path=self.output_path,
            batch_size=self.batch_size,
            embed_dim=self.embed_dim,
            projector=self.projector,
        )


class PredictionTask(luigi.Task, SharedParamsMixin):
    num_neighbors = luigi.IntParameter(
        default=1,
        description="Number of neighbors for KNN prediction.",
    )

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/_SUCCESS")

    def run(self):
        run_prediction(
            metadata_path=self.metadata_path,
            embedding_path=self.embedding_path,
            output_path=self.output_path,
            num_neighbors=self.num_neighbors,
        )


class Workflow(luigi.Task):
    def run(self):
        root = Path("~/scratch/animalclef").expanduser()
        metadata_path = root / "raw/metadata.csv"
        embedding_path = root / "processed/embeddings.parquet"

        # let's project a bunch of data
        tasks = []
        for model in (root / "models").glob("**/*.pt"):
            model_path = model.resolve()
            output_path = (
                root
                / "processed"
                / "embeddings"
                / model_path.parent.name
                / model_path.stem
            )
            output_path.mkdir(parents=True, exist_ok=True)

            kwargs = (
                dict(embed_dim=256, projector="nonlinear")
                if "nonlinear" in model_path.as_posix()
                else dict(embed_dim=128, projector="linear")
            )
            task = EmbeddingTask(
                metadata_path=metadata_path.as_posix(),
                embedding_path=embedding_path.as_posix(),
                projection_head_path=model_path.as_posix(),
                output_path=output_path.as_posix(),
                **kwargs,
            )
            tasks.append(task)
        yield tasks

        tasks = []
        for embedding in (root / "processed" / "embeddings").glob(
            "**/embeddings.parquet"
        ):
            embedding_path = embedding.resolve()
            output_path = (
                root
                / "processed"
                / "submissions"
                / embedding_path.parent.parent.name
                / embedding_path.parent.name
            )
            task = PredictionTask(
                metadata_path=metadata_path.as_posix(),
                embedding_path=embedding_path.as_posix(),
                output_path=output_path.as_posix(),
            )
            tasks.append(task)
        yield tasks


@app.command()
def main(workers=4):
    luigi.build([Workflow()], workers=workers, local_scheduler=True)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    app()
