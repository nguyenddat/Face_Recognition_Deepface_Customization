from abc import ABC, abstractmethod
from typing import *
import numpy as np

from helpers import package_helpers

tf_version = package_helpers.get_tf_major_version()
if tf_version == 1:
    from keras.models import Model
else:
    from tensorflow.keras.models import Model
    
class Demography(ABC):
    model: Model
    model_name: str 
    
    @abstractmethod
    def predict(self, img: np.ndarray) -> Union[np.ndarray, np.float64]:
        pass