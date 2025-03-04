from functools import cache

import numpy as np
import torch

from pyspark.ml import Transformer
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, FloatType, StructField, StructType

from animalclef.params import HasModelParamsMixin
from animalclef.serde import deserialize_image


@cache
def get_dino_processor_and_model(model_name: str) -> tuple:
    """Return processor and model."""
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    return processor, model


class WrappedDino(Transformer, HasModelParamsMixin):
    """
    Wrapper for fine-tuned DINOv2 to add it to the pipeline.
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        model_name: str = "facebook/dinov2-base",
        batch_size: int = 32,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            modelName=model_name,
            batchSize=batch_size,
        )

    def _nvidia_smi(self):
        from subprocess import run

        try:
            run(["nvidia-smi"], check=True)
        except Exception:
            pass

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""

        # check on the nvidia stats when generating the predict function
        self._nvidia_smi()
        processor, model = get_dino_processor_and_model(self.getModelName())

        def predict(inputs: np.ndarray) -> np.ndarray:
            # extract [CLS] token embeddings
            with torch.no_grad():
                outputs = model(
                    **processor(
                        images=[deserialize_image(img) for img in inputs],
                        return_tensors="pt",
                    )
                )
                features = outputs.last_hidden_state
                cls_token = features[:, 0, :]
                avg_patch_token = features[:, 1:, :].mean(dim=1)

            return {
                "cls": cls_token.cpu().numpy(),
                "avg_patch": avg_patch_token.cpu().numpy(),
            }

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=StructType(
                    [
                        StructField("cls", ArrayType(FloatType())),
                        StructField("avg_patch", ArrayType(FloatType())),
                    ]
                ),
                batch_size=self.getBatchSize(),
            )(self.getInputCol()),
        )
