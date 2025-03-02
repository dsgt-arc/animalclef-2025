import numpy as np
from pyspark.ml import Transformer
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, FloatType

from animalclef.params import HasModelParamsMixin
from animalclef.serde import deserialize_image


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
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        # check on the nvidia stats when generating the predict function
        self._nvidia_smi()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoFeatureExtractor.from_pretrained(self.getModelName())
        model = AutoModel.from_pretrained(self.getModelName()).to(device)

        def predict(inputs: np.ndarray) -> np.ndarray:
            images = processor(
                images=[deserialize_image(img) for img in inputs],
                return_tensors="pt",
            ).pixel_values

            # extract [CLS] token embeddings
            with torch.no_grad():
                features = model.forward_features(torch.stack(images))
                cls_token = features[:, 0, :]

            return cls_token.cpu().numpy()

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=ArrayType(FloatType()),
                batch_size=self.getBatchSize(),
            )(self.getInputCol()),
        )
