import pytest
from animalclef.embed.transform import WrappedDino
import imageio.v3 as iio
from animalclef.serde import serialize_image
import numpy as np


@pytest.fixture
def image_dataset(spark):
    img = iio.imread("imageio:astronaut.png")
    return spark.createDataFrame(
        [
            {"image_id": "1", "content": serialize_image(img)},
            {"image_id": "2", "content": serialize_image(img)},
        ],
    )


def test_wrapped_dino(image_dataset):
    wrapped_dino = WrappedDino(input_col="content", output_col="token", batch_size=1)
    df = wrapped_dino.transform(image_dataset)
    assert df.count() == 2
    assert df.select("image_id", "token.cls", "token.avg_patch").columns == [
        "image_id",
        "cls",
        "avg_patch",
    ]
    row = df.first()
    assert np.array(row.token.cls).shape == (768,)
    assert np.array(row.token.avg_patch).shape == (768,)
