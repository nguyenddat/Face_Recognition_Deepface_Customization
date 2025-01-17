from abc import ABC 
from typing import *
import numpy as np

from ..helpers import package_helpers

tf_version = package_helpers.get_tf_major_version()
if tf_version == 2:
    from tensorflow.keras.models import Model
else:
    from keras.models import Model
    
class FacialRecogition(ABC):
    model: Union[Model, Any]
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int
    
    def forward(self, img: np.ndarray) -> List[float]:
        if not isinstance(self.model, Model):
            raise ValueError("Must overwrite forward method if it is not a keras model")
        
        return self.model(img, training = False).numpy()[0].tolist()