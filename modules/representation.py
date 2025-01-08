from typing import *

import numpy as np
from heapq import nlargest

from  helpers import image_helpers
from modules import modeling, detection, preprocessing
from schemas.FacialRecognition import FacialRecogition

def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    max_faces: Optional[int] = None
) -> List[Dict[str, Any]]:
    resp_objs = []

    model: FacialRecogition = modeling.build_model(
        task = "facial_recognition",
        model_name = model_name
    )
    
    target_size = model.input_shape
    if detector_backend != "skip":
        img_objs = detection.extract_faces(
            img_path = img_path,
            detector_backend = detector_backend,
            align = align,
            expand_percentage = expand_percentage,
            max_faces = max_faces
        )
    else:
        img, _ = image_helpers.load_image(img_path)

        if len(img.shape) != 3:
            raise ValueError("Input img must be 3 dimensional")
        
        img_objs = [
            {
                "face": img,
                "facial_area": {
                    "x": 0, "y": 0,
                    "w": img.shape[0], "h": img.shape[1]
                },
                "confidence": 0
            }
        ]
        
    if max_faces is not None and max_faces < len(img_objs):
        img_objs = nlargest(
            max_faces, img_objs, key = lambda img_obj: img_obj["facial_area"]["w"] * img_obj["facial_area"]["h"]
        )

    for img_obj in img_objs:
        img = img_obj["face"]
        img = img[:, :, ::-1]

        region = img_obj["facial_area"]
        confidence = img_obj["confidence"]

        img = preprocessing.resize_image(
            img = img,
            target_size = (target_size[1], target_size[0])
        )
        
        img = preprocessing.normalize_input(img = img, normalization = normalization)

        embedding = model.forward(img)

        resp_objs.append(
            {
                "embedding": embedding,
                "facial_area": region,
                "face_confidence": confidence
            }
        )
    return resp_objs