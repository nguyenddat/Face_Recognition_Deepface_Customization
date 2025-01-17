import os
from typing import *

import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..helpers import image_helpers
from ..modules import detection, verification

def concurrent_load_store_data(dir_path, label, backup_file):
    X = []
    y = []
    try:
        if os.path.exists(backup_file):
            with open(backup_file, "rb") as file:
                backup_data = pickle.load(file)
            X.extend(backup_file)
            y.extend([label] * len(backup_data))
        else:
            current_data = []
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path) and file.endswith(".jpg"):
                    img_embeddings, _ = detection.extract_embeddings_and_facial_areas(
                        img_path = file_path
                    )
                    current_data.extend(img_embeddings)
            
            with open(backup_file, "wb") as file:
                pickle.dump(current_data, file)
            
            X.extend(current_data)
            y.extend([label] * len(current_data))
    
    except Exception as err:
        raise SystemError("Error while loading stored data") from err

    return X, y

    
def load_store_data(
    model_name: str,
    detector_backend: str, 
    db_path: str
):
    if not os.path.isdir(db_path):
        raise ValueError("db path is not existed")
    
    file_parts = ["ds", "model", model_name, "detector", detector_backend]
    file_name = "_".join(file_parts).lower() + ".pkl"
    datastore_path = os.path.join(db_path, file_name)
    
    if os.path.exists(datastore_path):
        with open(datastore_path, "rb") as file:
            return pickle.load(file)
    
    representations = {"X": [], "y": []}

    for dir in os.scandir(db_path):
        if os.path.isdir(dir):
            label = dir.name
            dir_path = dir.path
            backup_file = os.path.join(dir_path, "backup.pkl")
            
            X, y = concurrent_load_store_data(dir_path, label, backup_file)
            print(X)
            print(y)
            representations["X"].extend(X)
            representations["y"].extend(y)
    
    with open(datastore_path, "wb") as file:
        pickle.dump(representations, file)
    
    return representations
            
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
    representations = load_store_data(
        model_name = model_name,
        detector_backend = detector_backend,
        db_path = db_path
    )
    
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
    
    
    
    