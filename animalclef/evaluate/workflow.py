import luigi
import typer
from .embed import get_embeddings
from pathlib import Path

app = typer.Typer()


class EmbeddingTask(luigi.Task):
    metadata_path = luigi.Parameter(
        description="Path to the metadata file containing image IDs and embeddings."
    )
    embedding_path = luigi.Parameter(
        description="Path to the file containing the embeddings."
    )
    projection_head_path = luigi.Parameter(
        description="Path to the projection head model file."
    )
    output_path = luigi.Parameter(
        description="Path to the output file where the embeddings will be saved."
    )
    batch_size = luigi.IntParameter(
        default=32,
        description="Batch size for loading the embeddings.",
    )
    embed_dim = luigi.IntParameter(
        default=128,
        description="Dimension of the embeddings.",
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
                / "submission"
                / model_path.parent.name
                / model_path.stem
            )
            output_path.mkdir(parents=True, exist_ok=True)

            task = EmbeddingTask(
                metadata_path=metadata_path.as_posix(),
                embedding_path=embedding_path.as_posix(),
                projection_head_path=model_path.as_posix(),
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
