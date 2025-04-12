import torch
import pytest
from torch.utils.data import DataLoader
from animalclef.train.triplet.datamodule import AnimalTripletDataModule


@pytest.mark.parametrize(
    "use_grid, grid_size",
    [
        (False, 2),  # No grid
        (True, 2),  # Grid size 2
    ],
)
def test_plant_dataset(pandas_df, use_grid, grid_size):
    dataset = PlantDataset(
        pandas_df,
        transform=None,
        use_grid=use_grid,
        grid_size=grid_size,
    )
    assert len(dataset) == 2
    sample_data = dataset[0]
    assert isinstance(sample_data, torch.Tensor)
    expected_shape = (
        (grid_size**2, *sample_data.shape[1:]) if use_grid else sample_data.shape
    )
    assert sample_data.shape == expected_shape