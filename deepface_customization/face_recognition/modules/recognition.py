import os
from typing import *

import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..modules import detection, verification

def find(
    img_path: Union[str, np.ndarray],
    data: dict,
    distance_metric: str = "cosine",
    detector_backend: str = "opencv",
    threshold: Optional[float] = None
):
    
    if len(data["X"]) == 0 or len(data["y"]) == 0:    
        return []
    
    # ______________________________________________________________________
    source_objs = detection.extract_faces(
        img_path = img_path,
        detector_backend = detector_backend
    )
    
    def concurrent_recognize(source_obj):
        return verification.recognize(
            img = source_obj,
            data_store = data, 
            threshold = threshold, 
            distance_metric = distance_metric
        )
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(concurrent_recognize, source_objs))
    
    return results
    
    
    
    