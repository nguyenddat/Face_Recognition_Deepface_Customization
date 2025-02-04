from typing import *

import numpy as np

from ..helpers import data_helpers
from face_recognition.modules import recognition

class FaceRecognition:
    def __init__(self):
        self.data = data_helpers.load_stored_data()
    
    def find(
            img_path: Union[str, np.ndarray],
            data: dict,
            distance_metric: str = "cosine",
            detector_backend: str = "opencv",
            threshold: Optional[float] = None
    ):
        return recognition.find(
            img_path = img_path,
            data = data,
            distance_metric = distance_metric,
            detector_backend = detector_backend,
            threshold = threshold
        )

face_recognition = FaceRecognition()