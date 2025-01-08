import os 
from typing import *
import numpy as np

import cv2

from schemas.Detector import Detector, FacialAreaRegion

class OpenCvClient(Detector):
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self):
        detector = {}
        detector["face_detector"] = self.__build_cascade("haarcascade")
        detector["eye_detector"] = self.__build_cascade("haarcascade_eye")
        return detector
    
    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        resp = []
        detected_face = None
        
        try:
            faces, _, scores = self.model["face_detector"].detectMultiScale3(
                img, 1.1, 10, outputRejectLevels = True
            )
        except:
            pass
        
        if len(faces) > 0:
            for (x, y, w, h), confidence in zip(faces, scores):
                detected_face = img[int(y): int(y + h), int(x): int(x + w)]
                left_eye, right_eye = self.find_eyes(img = detected_face)
                
                if left_eye is not None:
                    left_eye = (int(x + left_eye[0]), int(y + left_eye[1]))
                if right_eye is not None:
                    right_eye = (int(x + right_eye[0]), int(y + right_eye[1]))
                
                facial_area = FacialAreaRegion(
                    x = x, y = y, w = w, h = h,
                    left_eye = left_eye,
                    right_eye = right_eye,
                    confidence = (100 - confidence) / 100
                )
                resp.append(facial_area)

        return resp
    
    def find_eyes(self, img: np.ndarray) -> tuple:
        left_eye = None
        right_eye = None
        
        if img.shape[0] == 0 or img.shape[1] == 0:
            return left_eye, right_eye
        
        detected_face_gray = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY
        )
        
        eyes = self.model["eye_detector"].detectMultiScale(detected_face_gray, 1.1, 10)
        eyes = sorted(eyes, key = lambda x: abs(x[2] * x[3]), reverse = True)
        if len(eyes) >= 2:
            eye_1 = eyes[0]
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                right_eye = eye_1
                left_eye = eye_2
            else:
                right_eye = eye_2
                left_eye = eye_1
            right_eye = (
                int(right_eye[0] + (right_eye[2] / 2)),
                int(right_eye[1] + (right_eye[3] / 2))
            )
            left_eye = (
                int(left_eye[0] + (left_eye[2] / 2)),
                int(left_eye[1] + (left_eye[3] / 2))
            )
        return left_eye, right_eye
    
    def __build_cascade(self, model_name = "haarcascade") -> Any:
        opencv_path = self.__get_opencv_path()
        if model_name == "haarcascade":
            face_detector_path = os.path.join(opencv_path, "haarcascade_frontalface_default.xml")
            if not os.path.isfile(face_detector_path):
                raise ValueError("Confirm that opencv is installed on your environment! Expected path ",
                    face_detector_path, " violated")
            detector = cv2.CascadeClassifier(face_detector_path)
        
        elif model_name == "haarcascade_eye":
            eye_detector_path = os.path.join(opencv_path, "haarcascade_eye.xml")
            if not os.path.isfile(eye_detector_path):
                raise ValueError("Confirm that opencv is installed on your environment! Expected path ",
                    face_detector_path, " violated")
            detector = cv2.CascadeClassifier(eye_detector_path)
        
        else:
            raise ValueError(f"unimplemented model_name for build_cascade - {model_name}")

        return detector
    
    def __get_opencv_path(self) -> str:
        return os.path.join(os.path.dirname(cv2.__file__), "data")