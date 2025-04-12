import pytest
from animalclef.spark import spark_resource
import imageio.v3 as iio
from animalclef.serde import serialize_image
import os

import io
import pandas as pd
from PIL import Image


os.environ["PYSPARK_DRIVER_CORES"] = "2"
os.environ["PYSPARK_DRIVER_MEMORY"] = "4g"


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("spark_data")
    with spark_resource(local_dir=tmp_path.as_posix()) as spark:
        yield spark


@pytest.fixture
def image_df(spark):
    img = iio.imread("imageio:astronaut.png")
    return spark.createDataFrame(
        [
            {"image_id": "1", "content": serialize_image(img)},
            {"image_id": "2", "content": serialize_image(img)},
        ],
    )


@pytest.fixture
def image_df_path(image_df, tmp_path):
    path = tmp_path / "image_dataset"
    image_df.write.parquet(str(path))
    return path

@pytest.fixture
def pandas_df():
    # generate a small dummy image(RGB, 32X32) for testing
    img = Image.new("RGB", (32, 32), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()
    data = {"data": [img_bytes, img_bytes]}

    return pd.DataFrame(data)