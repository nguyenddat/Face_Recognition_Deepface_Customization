from typing import *

import numpy as np
import cv2
from heapq import nlargest

from ..modules import modeling, preprocessing, representation
from ..helpers import image_helpers
from ..schemas.Detector import *

def extract_faces(img_path: Union[str, np.ndarray],
                  detector_backend: str = "opencv",
                  align: bool = True,
                  normalize_face: bool = True):
    resp_objs = []
    
    img, img_name = image_helpers.load_image(img_path)
    if img is None:
        raise ValueError(f"Exception while loading image: {img_name}")
    
    height, width, _ = img.shape
    
    base_region = FacialAreaRegion(
        x = 0, y = 0, w = width, h = height,
        confidence = 0
    )
    
    face_objs = detect_faces(img = img,
                             detector_backend = detector_backend, 
                             align = align)
    
    if len(face_objs) == 0:
        face_objs = [DetectedFace(img = img, facial_area = base_region, confidence = 0)]
    
    for face_obj in face_objs:
        current_img = face_obj.img
        current_region = face_obj.facial_area

        if current_img.shape[0] == 0 or current_img.shape[1] == 0:
            continue
        
        if normalize_face:
            current_img = current_img / 255
        
        x = max(0, int(current_region.x))
        y = max(0, int(current_region.y))
        w = min(width - x - 1, int(current_region.w))
        h = min(height - y - 1, int(current_region.h))

        facial_area = {"x": x, "y": y, "w": w, "h": h}
        if current_region.left_eye:
            facial_area["left_eye"] = current_region.left_eye
        if current_region.right_eye:
            facial_area["right_eye"] = current_region.right_eye
        if current_region.nose:
            facial_area["nose"] = current_region.nose
        if current_region.left_mouth:
            facial_area["left_mouth"] = current_region.left_mouth
        if current_region.right_mouth:
            facial_area["right_mouth"] = current_region.right_mouth
        
        resp_objs.append({"img": current_img,
                          "facial_area": facial_area,
                          "confidence": round(float(current_region.confidence or 0), 2)})
    return resp_objs

def detect_faces(img: np.ndarray,
                 detector_backend: str = "opencv",
                 align: bool = True) -> List[DetectedFace]:
    height, width, _ = img.shape
    
    face_detector: Detector = modeling.build_model(
        task = "face_detector",
        model_name = detector_backend
    )
    
    height_border = int(0.5 * height)
    width_border = int(0.5 * width)

    if align:
        img = cv2.copyMakeBorder(
            src = img,
            top = height_border,
            bottom = height_border,
            left = width_border,
            right = width_border,
            borderType = cv2.BORDER_CONSTANT,
            value = [0, 0, 0]
        )
    
    facial_areas = face_detector.detect_faces(img)
    
    return [
        extract_face(img = img,
                     facial_area = facial_area,
                     align = align,
                     width_border = width_border,
                     height_border = height_border)
        for facial_area in facial_areas
    ]
    
def extract_face(img: np.ndarray,
                 facial_area: FacialAreaRegion,
                 align: bool,
                 width_border: int,
                 height_border: int) -> DetectedFace:
    x, y, w, h = facial_area.x, facial_area.y, facial_area.w, facial_area.h
    left_eye, right_eye = facial_area.left_eye, facial_area.right_eye
    nose = facial_area.nose
    left_mouth, right_mouth = facial_area.left_mouth, facial_area.right_mouth
    confidence = facial_area.confidence
    
    detected_face = img[
        int(y): int(y + h),
        int(x): int(x + w)
    ]
    
    if align:
        sub_img, relative_x, relative_y = preprocessing.extract_sub_img(img = img,
                                                            facial_area = (x, y, w, h))
        
        aligned_sub_img, angle = preprocessing.align_img_with_eyes(img = sub_img,
                                              left_eye = left_eye,
                                              right_eye = right_eye)
        
        rotated_x1, rotated_y1, rotated_x2, rotated_y2 = preprocessing.project_facial_area(facial_area = (relative_x,
                                                                                            relative_y,
                                                                                            relative_x + w,
                                                                                            relative_y + h),
                                                                             angle = angle,
                                                                             size = (sub_img.shape[0], sub_img.shape[1]))
        
        detected_face = aligned_sub_img[
            int(rotated_y1): int(rotated_y2),
            int(rotated_x1): int(rotated_x2)
        ]
        
        x = x - width_border
        y = y - height_border
        
        if left_eye is not None:
            left_eye = (left_eye[0] - width_border, left_eye[1] - height_border)            
        if right_eye is not None:
            right_eye = (right_eye[0] - width_border, right_eye[1] - height_border)            
        if nose is not None:
            nose = (nose[0] - width_border, nose[1] - height_border)            
        if left_mouth is not None:
            left_mouth = (left_mouth[0] - width_border, left_mouth[1] - height_border)            
        if right_mouth is not None:
            right_mouth = (right_mouth[0] - width_border, right_mouth[1] - height_border)            
        
    return DetectedFace(
        img = detected_face,
        facial_area = FacialAreaRegion(
            x = x, y = y, w = w, h = h,
            confidence = confidence,
            left_eye = left_eye,
            right_eye = right_eye,
            nose = nose,
            left_mouth = left_mouth,
            right_mouth = right_mouth
        ),
        confidence = confidence or 0
    )

def extract_embeddings_and_facial_areas(img_path: Union[str, np.ndarray],
                                        model_name: str = "VGG-Face",
                                        detector_backend: str = "opencv",
                                        align: bool = True,
                                        normalize_face: bool = True):
    try:
        img_embeddings, img_facial_areas =  extract_faces_and_embeddings(img_path = img_path,
                                                                         model_name = model_name,
                                                                         detector_backend = detector_backend,
                                                                         align = align,
                                                                         normalize_face = normalize_face)
    except ValueError as err:
        raise ValueError("Exception while processing img") from err
    
    return img_embeddings, img_facial_areas

def extract_faces_and_embeddings(img_path: np.ndarray,
                                 model_name: str = "VGG-Face",
                                 detector_backend: str = "opencv",
                                 align: bool = True,
                                 normalize_face: bool = True):
    embeddings = []
    facial_areas = []

    resp_objs = extract_faces(img_path = img_path,
                              detector_backend = detector_backend,
                              align = align, 
                              normalize_face = normalize_face)
    
    for resp_obj in resp_objs:
        current_img = resp_obj["img"]
        
        img_embedding = representation.represent(img = current_img, model_name = model_name)
        embeddings.append(img_embedding)
        facial_areas.append(resp_obj["facial_area"])
    
    return embeddings, facial_areas