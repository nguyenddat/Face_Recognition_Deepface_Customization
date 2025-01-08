from typing import *

import numpy as np

from modules import representation, detection, modeling
from schemas.FacialRecognition import FacialRecogition
from helpers.logger import logger
from helpers import model_helpers

def verify(
    img1_path: Union[str, np.ndarray, List[float]],
    img2_path: Union[str, np.ndarray, List[float]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    threshold: Optional[float] = None,
) -> Dict[Any, Any]:
    model: FacialRecogition = modeling.build_model(
        task = "facial_recognition", model_name = model_name
    )
    
    dims = model.output_shape
    
    no_facial_area = {
        "x": None, "y": None, "w": None, "h": None, 
        "left_eye": None, "right_eye": None
    }
    
    def extract_embeddings_and_facial_areas(
        img_path: Union[str, np.ndarray, List[float]]
    ):
        if isinstance(img_path, list):
            if not all(isinstance(dim, float) for dim in img_path):
                raise ValueError("Ensure all value in img_path is float")
            
            img_embeddings = [img_path]
            img_facial_areas = [no_facial_area]
        else:
            # try:
            img_embeddings, img_facial_areas = __extract_faces_and_embeddings(
                img_path = img_path,
                model_name = model_name,
                detector_backend = detector_backend,
                align = align,
                expand_percentage = expand_percentage,
                normalization = normalization
            )
            # except Exception as err:
            #     raise ValueError("Exception while processing img_path") from err
        return img_embeddings, img_facial_areas
    
    img1_embeddings, img1_facial_areas = extract_embeddings_and_facial_areas(img1_path)
    img2_embeddings, img2_facial_areas = extract_embeddings_and_facial_areas(img2_path)

    min_distance, min_idx, min_idy = float("inf"), None, None
    for idx, img1_embedding in enumerate(img1_embeddings):
        for idy, img2_embedding in enumerate(img2_embeddings):
            distance = model_helpers.find_distance(img1_embedding, img2_embedding, distance_metric)
            if distance < min_distance:
                min_distance, min_idx, min_idy = distance, idx, idy

    threshold = threshold or model_helpers.find_threshold(model_name, distance_metric)
    distance = float(min_distance)
    facial_areas = {
        no_facial_area if min_idx is None else img1_facial_areas[min_idx],
        no_facial_area if min_idy is None else img2_facial_areas[min_idy],
    }
    
    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "detector_backend": detector_backend,
        "facial_areas": {
            "img1": facial_areas[0],
            "img2": facial_areas[1],
        }
    }
    
    return resp_obj

def __extract_faces_and_embeddings(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base"
) -> Tuple[List[List[float]], List[dict]]:
    embeddings = []
    facial_areas = []

    img_objs = detection.extract_faces(
        img_path = img_path,
        detector_backend = detector_backend,
        align = align,
        expand_percentage = expand_percentage,
        normalize_face = normalization
    )
    
    for img_obj in img_objs:
        img_embedding_obj = representation.represent(
            img_path = img_obj["face"],
            model_name = model_name,
            detector_backend = "skip",
            align = align,
            normalization = normalization
        )
        img_embedding = img_embedding_obj[0]["embedding"]
        embeddings.append(img_embedding)
        facial_areas.append(img_obj["facial_area"])

    return embeddings, facial_areas