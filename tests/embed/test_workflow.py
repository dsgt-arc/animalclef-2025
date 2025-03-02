from animalclef.embed.workflow import embed_dinov2
import numpy as np


def test_embed_dino_v2(spark, image_df_path, tmp_path):
    embed_dinov2(
        input_path=str(image_df_path),
        output_path=str(tmp_path / "output"),
        batch_size=1,
    )
    df = spark.read.parquet(str(tmp_path / "output"))
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
