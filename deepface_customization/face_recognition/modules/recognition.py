import os
from typing import *

import pickle
import numpy as np

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
    resp_objs = []

    source_objs = detection.extract_faces(
        img_path = img_path,
        detector_backend = detector_backend
    )
    
    for source_obj in source_objs:
        img_embeddings, facial_areas = detection.extract_embeddings_and_facial_areas(source_obj["img"], detector_backend = detector_backend)
        
        for img_embedding, facial_area in zip(img_embeddings, facial_areas):
            resp_obj = verification.recognize(
                img = img_embedding,
                data_store = data,
                threshold = threshold,
                distance_metric = distance_metric
            )

            resp_objs.append({"prediction": resp_obj, "facial_area": facial_area})
    
    return resp_objs
