from typing import *

import numpy as np
from retinaface import RetinaFace as rf

from schemas.Detector import Detector, FacialAreaRegion

class RetinaFaceClient(Detector):
    def __init__(self):
        self.model = rf.build_model()
    
    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        resp = []

        obj = rf.detect_faces(img, model = self.model, threshold=0.9)

        if not isinstance(obj, dict):
            return resp
        
        for face_idx, identity in obj.items():
            detection = identity["facial_area"]

            y = detection[1]
            h = detection[3] - y
            x = detection[0]
            w = detection[2] - x
            
            left_eye = identity["landmarks"]["left_eye"]
            right_eye = identity["landmarks"]["right_eye"]
            nose = identity["landmarks"]["nose"]
            right_mouth = identity["landmarks"]["mouth_right"]
            left_mouth = identity["landmarks"]["mouth_left"]

            left_eye = tuple(int(i) for i in left_eye)
            right_eye = tuple(int(i) for i in right_eye)

            if nose is not None:
                nose = tuple(int(i) for i in nose)
            if right_mouth is not None:
                right_mouth = tuple(int(i) for i in right_mouth)
            if left_mouth is not None:
                left_mouth = tuple(int(i) for i in left_mouth)
            
            confidence = identity["score"]

            facial_area = FacialAreaRegion(
                x = x, y = y, w = w, h = h,
                left_eye = left_eye,
                right_eye = right_eye,
                confidence = confidence,
                nose = nose,
                right_mouth = right_mouth,
                left_mouth = left_mouth
            )
            
            resp.append(facial_area)
        return resp
            