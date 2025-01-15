import os
from typing import *

import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from helpers import image_helpers
from modules import detection, verification

def find(
    img_path: Union[str, np.ndarray],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    detector_backend: str = "opencv",
    align: bool = True,
    threshold: Optional[float] = None,
    normalize_face: bool = True,
):
    if not os.path.isdir(db_path):
        raise ValueError("db path is not existed")
    
    file_parts = ["ds", "model", model_name, "detector", detector_backend]
    file_name = "_".join(file_parts) + ".pkl"
    file_name = file_name.replace("_", "").lower()
    
    datastore_path = os.path.join(db_path, file_name)
    representations = {"X": [], "y": []}
    
    if not os.path.exists(datastore_path):
        with open(datastore_path, "wb") as file:
            pickle.dump(representations, file, pickle.HIGHEST_PROTOCOL)

    with open(datastore_path, "rb") as file:
        representations = pickle.load(file)
    
    storage_images = set(image_helpers.yield_image(path = db_path))
    if len(storage_images) == 0:
        raise ValueError(f"No item found in {db_path}")
    
    if len(representations["X"]) == 0 or len(representations["y"]) == 0:    
        return []
    
    # ______________________________________________________________________
    source_objs = detection.extract_faces(
        img_path = img_path,
        detector_backend = detector_backend,
        align = align,
        normalize_face = normalize_face
    )
    
    def concurrent_recognize(source_obj):
        return verification.recognize(
            img = source_obj,
            data_store = representations, 
            threshold = threshold, 
            distance_metric = distance_metric
        )
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(concurrent_recognize, source_objs))
    
    return results
    
    
    
    