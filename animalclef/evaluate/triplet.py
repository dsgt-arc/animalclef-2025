import functools
import operator
import random

import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from .model import LinearProjectionHead, NonlinearProjectionHead, MRLTripletLoss


def split_reid_data(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.5,
    group_col: str = "identity",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training, validation, and test sets for open-set re-identification tasks.
    Simplified method to get a training/validation split
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

    return (
        df[df[group_col].isin(train_ids)],
        df[df[group_col].isin(val_unknown_ids)],
        df[df[group_col].isin(test_unknown_ids)],
    )


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
    ):
        self.metadata = self._filter_metadata(metadata)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.metadata.identity)

    def _filter_metadata(self, metadata):
        # only do triplet learning on the database images that have at least 2 images
        metadata = metadata.loc[metadata["split"] == "database"]
        dfs = []
        for identity in metadata.identity.unique():
            id_metadata = metadata.loc[metadata.identity == identity].copy()
            if id_metadata.shape[0] >= 2:
                dfs.append(id_metadata)
        return pd.concat(dfs, axis=0).reset_index()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple:
        row = self.metadata.iloc[idx]
        embedding = torch.from_numpy(row.embeddings).to(self.device)
        idx = torch.tensor(idx).to(self.device)
        label = torch.tensor(self.label_encoder.transform([row.identity])[0]).to(
            self.device
        )
        return embedding, idx, label

    @functools.cache
    def _filter_positive(self, idx):
        df = self.metadata
        anchor = df.iloc[idx]
        # find all the rows that match the condition and sample 1
        cond = [(df.identity == anchor.identity), (df.image_id != anchor.image_id)]
        # we only need the index from these so we try to reduce the cache
        return df[functools.reduce(operator.__and__, cond)].index

    @functools.cache
    def _filter_negative(self, idx, same_species=False):
        """find a random row not in the current identity"""
        df = self.metadata
        anchor = df.iloc[idx]
        cond = [(df.identity != anchor.identity)]
        if same_species:
            cond.append((df.dataset == anchor.dataset))
        # we only need the index from these so we try to reduce the cache
        return df[functools.reduce(operator.__and__, cond)].index

    def sample_positive(self, idx, n=1):
        """find a row with the same label but different index"""
        return random.choices(self._filter_positive(idx), k=n)

    def sample_negative(self, idx, n=1, same_species=False):
        """find a random row not in the current identity"""
        return random.choices(self._filter_negative(idx), k=n)


def train_batch_semi_hard_negative(
    metadata,
    projection_head,
    output_path=None,
    batch_size=200,
    epochs=100,
    margin=1.0,
    nested_dims=[],
    loss_weights=None,
    learning_rate=5e-4,
    warmup_epochs=10,
    triplets_per_anchor=1,
):
    train_df, val_df, _ = split_reid_data(metadata)

    metadatas = {
        "train": train_df,
        "val": val_df,
    }
    datasets = {}
    dataloaders = {}
    for step in ["train", "val"]:
        datasets[step] = EmbeddingDataset(metadatas[step])
        dataloaders[step] = DataLoader(
            datasets[step],
            batch_size=batch_size,
            sampler=RandomSampler(datasets[step]),
        )

    optimizer = optim.Adam(projection_head.parameters(), lr=learning_rate)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    criterion = MRLTripletLoss(
        margin=margin, nested_dims=nested_dims, loss_weights=loss_weights
    )

    for epoch in tqdm(range(epochs)):
        for step in ["train", "val"]:
            if step == "train":
                projection_head.train()
            else:
                projection_head.eval()
            dataset = datasets[step]
            dataloader = dataloaders[step]

            running_loss = 0.0
            batch_count = 0
            epoch_triplets = 0

            for batch_embeddings, batch_indices, batch_labels in dataloader:

                @functools.cache
                def _embed(idx):
                    emb, _, _ = dataset[idx]
                    return projection_head(emb).unsqueeze(0)

                batch_size = batch_embeddings.size(0)
                if batch_size <= 1:
                    continue

                batch_embeddings = projection_head(batch_embeddings)
                batch_pairwise_dist = torch.cdist(
                    batch_embeddings, batch_embeddings, p=2
                )

                # Process each anchor in the batch
                batch_loss = 0
                batch_triplets = 0

                for anchor_embedding, anchor_index, anchor_label, anchor_dist in zip(
                    batch_embeddings, batch_indices, batch_labels, batch_pairwise_dist
                ):
                    anchor_embedding = anchor_embedding.unsqueeze(0)
                    anchor_index = int(anchor_index)

                    # we're going to generate triplets in the following way
                    # first we get a random positive of indices
                    # then we do batch semi-hard mining by looking for everything within the margin
                    # skip if it doesn't exist
                    for _ in range(triplets_per_anchor):
                        pos_idx = dataset.sample_positive(anchor_index)[0]
                        pos_embedding = _embed(pos_idx)
                        # distance positive -> dp
                        dp = torch.pairwise_distance(anchor_embedding, pos_embedding)

                        # BATCH SEMI-HARD NEGATIVE MINING
                        # Distances from the current anchor (projected) to all other *projected* samples in THIS BATCH
                        # Mask for negatives within the current batch (different true label than anchor)

                        # Semi-hard condition mask
                        semi_hard_mask = (
                            # labels that dont match the anchor
                            (batch_labels != anchor_label)
                            # dp < dn < dp+margin
                            & (anchor_dist > dp)
                            & (anchor_dist < (dp + criterion.margin))
                        )

                        # skip because no semi-hard
                        if not torch.any(semi_hard_mask):
                            continue

                        # Get indices (within the current batch) of these semi-hard negatives
                        semi_hard_indices = torch.where(semi_hard_mask)[0]

                        # Select one semi-hard negative (e.g., the hardest of the semi-hard, or random)
                        # To get the "hardest" of the semi-hard (closest to anchor among semi-hard):
                        smallest_neg = torch.min(anchor_dist[semi_hard_indices])
                        # Find which batch index corresponds to this distance among the semi-hard ones
                        neg_indices = torch.where(
                            semi_hard_mask & (anchor_dist == smallest_neg)
                        )[0]

                        if neg_indices.numel() == 0:
                            continue
                        neg_idx = neg_indices[0]

                        neg_embedding = batch_embeddings[neg_idx].unsqueeze(0)
                        triplet_loss = criterion(
                            anchor_embedding, pos_embedding, neg_embedding
                        )
                        batch_loss += triplet_loss
                        batch_triplets += 1
                        epoch_triplets += 1

                # Skip batch if no valid triplets
                if batch_triplets == 0:
                    continue

                # Normalize loss by number of triplets
                batch_loss = batch_loss / batch_triplets
                if step == "train":
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                # Update statistics
                running_loss += batch_loss.item()
                batch_count += 1

            # Print epoch statistics
            if batch_count > 0:
                current_lr = optimizer.param_groups[0]["lr"]
                epoch_loss = running_loss / batch_count
                print(
                    f"Step {step}, Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Triplet Count: {epoch_triplets}, LR: {current_lr}"
                )
            else:
                print(
                    f"Step {step}, Epoch {epoch + 1}/{epochs}, No valid triplets found"
                )

        # done with loop so update scheduler
        scheduler.step()

        # save intermediate heads
        if epoch > 0 and epoch % 10 == 0 and output_path is not None:
            name = f"{output_path}/head_epoch={epoch:03d}.pt"
            print(f"saving {name}")
            torch.save(projection_head.state_dict(), name)

    return projection_head


def train(
    metadata_path: str,
    embedding_path: str,
    output_path: str,
    embed_dim: int = 128,
    projector: str = "linear",
):
    metadata = pd.read_csv(metadata_path)
    embeddings_df = pd.read_parquet(embedding_path)
    merged_df = pd.merge(metadata, embeddings_df, on="image_id", how="inner")
    print(merged_df.head(3))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = {"linear": LinearProjectionHead, "nonlinear": NonlinearProjectionHead}[
        projector
    ](768, embed_dim).to(device)

    dim = embed_dim
    head = train_batch_semi_hard_negative(
        merged_df,
        projection_head=head,
        output_path=output_path,
        epochs=100,
        nested_dims=[128, 64, 32, 16, 2],
        loss_weights=[1, 128 / dim, 64 / dim, 32 / dim, 16 / dim, 2 / dim],
        warmup_epochs=10,
        learning_rate=1e-3,
    )

    # write the output
    torch.save(head.state_dict(), f"{output_path}/head.pt")
