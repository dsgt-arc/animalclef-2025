import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pacmap
import matplotlib.pyplot as plt
from pathlib import Path
from .model import LinearProjectionHead, NonlinearProjectionHead
from typer import Typer

app = Typer()


class EmbeddingDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple:
        row = self.metadata.iloc[idx]
        return row.image_id, torch.from_numpy(row.embeddings).float().to(self.device)


def plot_embeddings(embeddings, df, title="triplet embeddings"):
    X = np.stack(embeddings.embeddings)
    # 1D embedding to see how the embedding is reshaped
    c = pacmap.PaCMAP(n_components=1).fit_transform(X)
    del X
    X = np.stack(df.embeddings)
    g = pacmap.PaCMAP().fit_transform(X)

    plt.scatter(g[:, 0], g[:, 1], s=1, alpha=0.5, c=c)
    plt.title(title)
    plt.colorbar()
    plt.legend()


@app.command()
def get_embeddings(
    metadata_path: str,
    embedding_path: str,
    projection_head_path: str,
    output_path: str,
    batch_size: int = 32,
    input_dim: int = 768,
    embed_dim: int = 128,
    projector: str = "linear",
):
    embeddings = pd.read_parquet(embedding_path)
    merged_df = pd.merge(
        pd.read_csv(metadata_path), embeddings, on="image_id", how="inner"
    ).sort_values("image_id")
    dataset = EmbeddingDataset(merged_df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = {"linear": LinearProjectionHead, "nonlinear": NonlinearProjectionHead}[
        projector
    ](input_dim, embed_dim).to(device)
    state_dict = torch.load(projection_head_path, map_location=device)
    head.load_state_dict(state_dict)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    res = []
    head.eval()
    with torch.no_grad():
        for batch_image_ids, batch_embeddings in tqdm(dataloader):
            out = head(batch_embeddings).detach().cpu().numpy()
            assert out.shape[1] == embed_dim, out.shape
            for image_id, embedding in zip(batch_image_ids, out):
                res.append({"image_id": int(image_id), "embeddings": embedding})

    df = pd.DataFrame(res).sort_values("image_id")
    print(df.head())
    df.to_parquet(f"{output_path}/embeddings.parquet", index=False)

    head_path = " ".join(Path(projection_head_path).parts[-2:])
    plot_embeddings(df, df, title=f"PaCMAP of triplet embeddings {head_path}")
    plt.savefig(f"{output_path}/embeddings.png")


if __name__ == "__main__":
    app()
