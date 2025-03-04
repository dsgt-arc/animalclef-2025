from animalclef.embed.transform import WrappedDino
import numpy as np


def test_wrapped_dino(image_df):
    wrapped_dino = WrappedDino(input_col="content", output_col="token", batch_size=1)
    df = wrapped_dino.transform(image_df)
    df.printSchema()
    assert df.count() == 2
    assert df.select("image_id", "token.cls", "token.avg_patch").columns == [
        "image_id",
        "cls",
        "avg_patch",
    ]
    row = df.first()
    assert np.array(row.token.cls).shape == (768,)
    assert np.array(row.token.avg_patch).shape == (768,)
