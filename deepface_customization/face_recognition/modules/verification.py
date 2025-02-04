from typing import *

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..modules import representation, detection, modeling
from ..schemas.FacialRecognition import FacialRecogition
from ..helpers.logger import logger
from ..helpers import model_helpers

def recognize(
    img: np.ndarray,
    data_store: Dict[AnyStr, List[Union[np.ndarray, AnyStr]]],
    threshold: float,
    distance_metric: str = "cosine",
):
    X = data_store["X"]
    y = data_store["y"]
    
    def concurrent_compute_distance(idx_embedding):
        idx, embedding = idx_embedding
        distance = model_helpers.find_distance(embedding, img, distance_metric)
        return (idx, distance)
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(concurrent_compute_distance, enumerate(X)))
    
    min_idx, min_distance = min(results, key = lambda x: x[1])
    return {
        "verified": min_distance <= threshold,
        "distance": min_distance,
        "threshold": threshold,
        "prediction": y[min_idx]
    }

def verify(
    img1_path: Union[str, np.ndarray, List[float]],
    img2_path: Union[str, np.ndarray, List[float]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    threshold: Optional[float] = None
) -> Dict[Any, Any]:

    resp_objs = []
    
    no_facial_area = {"x": None, "y": None, "w": None, "h": None, "left_eye": None, "right_eye": None}    
    img1_embeddings, img1_facial_areas = detection.extract_embeddings_and_facial_areas(
        img_path = img1_path,
        model_name = model_name,
        detector_backend = detector_backend
    )
        
    img2_embeddings, img2_facial_areas = detection.extract_embeddings_and_facial_areas(
        img_path = img2_path,
        model_name = model_name,
        detector_backend = detector_backend
    )
    threshold = threshold or model_helpers.find_threshold(model_name, distance_metric)
    
    for idx, img1_embedding in enumerate(img1_embeddings):
        min_distance, min_idy = float("inf"), None
        for idy, img2_embedding in enumerate(img2_embeddings):
            distance = model_helpers.find_distance(img1_embedding, img2_embedding, distance_metric)
            if distance < min_distance:
                min_distance, min_idy = distance, idy
        
        distance = float(min_distance)
    
        facial_areas = [
            img1_facial_areas[idx],
            no_facial_area if min_idy is None else img2_facial_areas[min_idy],
        ]
    
        resp_objs.append({
            "verified": distance <= threshold,
            "distance": distance,
            # "facial_areas": {
            #     "img1": facial_areas[0],
            #     "img2": facial_areas[1],
            # },
            "threshold": threshold
        })
    
    return resp_objs