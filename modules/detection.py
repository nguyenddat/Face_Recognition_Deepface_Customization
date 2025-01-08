from typing import *

import numpy as np
import cv2
from heapq import nlargest

from modules import modeling
from schemas.Detector import *
from helpers.logger import logger
from helpers import image_helpers

def extract_faces(
    img_path: Union[str, np.ndarray],
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalize_face: bool = True,
    max_faces: Optional[int] = None
) -> List[Dict[str, Any]]:
    resp_objs = []

    img, img_name = image_helpers.load_image(img_path)
    
    if img is None:
        raise ValueError(f"Exception while loading {img_name}")
    
    height, width, _ = img.shape

    base_region = FacialAreaRegion(
        x = 0, y = 0,
        w = width, h = height,
        confidence = 0
    )
    
    if detector_backend == "skip":
        face_objs = [DetectedFace(
            img = img,
            facial_area = base_region,
            confidence = 0
        )]
    else:
        face_objs = detect_faces(
            img = img,
            detector_backend = detector_backend,
            align = align,
            expand_percentage = expand_percentage,
            max_faces = max_faces
        )
    
    if len(face_objs) == 0:
        raise ValueError("Face could not be detected")
    
    for face_obj in face_objs:
        current_img = face_obj.img
        current_region = face_obj.facial_area
        
        if current_img.shape[0] == 0 or current_img.shape[1] == 0:
            continue
        
        if normalize_face:
            current_img = (1/255) * current_img
        
        x = max(0, int(current_region.x))
        y = max(0, int(current_region.y))
        w = min(width - x - 1, int(current_region.w))
        h = min(height - y - 1, int(current_region.h))

        facial_area = {
            "x": x, "y": y, "w": w, "h": h,
            "left_eye": current_region.left_eye,
            "right_eye": current_region.right_eye
        }
        
        if current_region.nose is not None:
            facial_area["nose"] = current_region.nose
        if current_region.left_mouth is not None:
            facial_area["left_mouth"] = current_region.left_mouth
        if current_region.right_mouth is not None:
            facial_area["right_mouth"] = current_region.right_mouth
            
        resp_obj = {
            "face": current_img,
            "facial_area": facial_area,
            "confidence": round(float(current_region.confidence or 0), 2)
        }
        
        resp_objs.append(resp_obj)
    
    if len(resp_objs) == 0:
        raise ValueError("Exception while extracting faces")
    
    return resp_objs

def detect_faces(
    img: np.ndarray,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    max_faces: Optional[int] = None,
) -> List[DetectedFace]:
    height, width, _ = img.shape
    
    face_detector: Detector = modeling.build_model(
        task = "face_detector", model_name = detector_backend
    )
    
    if expand_percentage < 0:
        logger.warn(f"Overwritten expand_percentage: {expand_percentage} to 0")
        expand_percentage = 0
        
    height_border = int(0.5 * height)
    width_border = int(0.5 * width)

    if align:
        img = cv2.copyMakeBorder(
            img,
            height_border,
            height_border,
            width_border,
            width_border,
            cv2.BORDER_CONSTANT,
            value = [0, 0, 0]
        )
    
    facial_areas = face_detector.detect_faces(img)
    
    if max_faces is not None and max_faces < len(facial_areas):
        facial_areas = nlargest(
            max_faces, facial_areas, key = lambda facial_area: facial_area.w * facial_area.h
        )
    
    return [
        extract_face(
            img = img,
            facial_area = facial_area,
            align = align,
            expand_percentage = expand_percentage,
            width_border = width_border,
            height_border = height_border
        )
        for facial_area in facial_areas
    ]
    
def extract_face(
    img: np.ndarray,
    facial_area: FacialAreaRegion,
    align: bool,
    expand_percentage: int,
    width_border: int,
    height_border: int
) -> DetectedFace:
    x, y, w, h = facial_area.x, facial_area.y, facial_area.w, facial_area.h
    left_eye = facial_area.left_eye
    right_eye = facial_area.right_eye
    confidence = facial_area.confidence
    nose = facial_area.nose
    left_mouth = facial_area.left_mouth
    right_mouth = facial_area.right_mouth

    if expand_percentage > 0:
        expanded_w = w + int(w * expand_percentage / 100)
        expanded_h = h + int(h * expand_percentage / 100)

        x = max(0, x - int((expanded_w - w) / 2))
        y = max(0, y - int((expanded_h - h) / 2))
        w = min(img.shape[1] - x, expanded_w)
        h = min(img.shape[0] - y, expanded_h)

    detected_face = img[int(y) : int(y + h), int(x): int(x + w)]
    if align:
        sub_img, relative_x, relative_y = extract_sub_image(img = img, facial_area = (x, y, w, h))

        aligned_sub_img, angle = align_img_wrt_eyes(
            img = sub_img, left_eye = left_eye, right_eye = right_eye
        )
        
        rotated_x1, rotated_y1, rotated_x2, rotated_y2 = project_facial_area(
            facial_area = (
                relative_x,
                relative_y,
                relative_x + w,
                relative_y + h
            ),
            angle = angle,
            size = (sub_img.shape[0], sub_img.shape[1])
        ) 
        
        detected_face = aligned_sub_img[
            int(rotated_y1) : int(rotated_y2), int(rotated_x1): int(rotated_x2)
        ]
        
        del aligned_sub_img, sub_img
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
            x = x, y = y, h = h, w = w,
            confidence = confidence,
            left_eye = left_eye,
            right_eye = right_eye,
            nose = nose,
            left_mouth = left_mouth,
            right_mouth = right_mouth,
        ),
        confidence = confidence or 0
    )    

def extract_sub_image(
    img: np.ndarray,
    facial_area: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, int, int]:
    x, y, w, h = facial_area
    relative_x = int(0.5 * w)
    relative_y = int(0.5 * h)

    x1, y1 = x - relative_x, y - relative_y
    x2, y2 = x + w + relative_x, y + h + relative_y

    if (x1 >= 0) and (y1 >= 0) and (x2 <= img.shape[1]) and (y2 <= img.shape[0]):
        return img[y1: y2, x1: x2], relative_x, relative_y
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    cropped_region = img[y1:y2, x1:x2]

    extracted_face = np.zeros(
        (h + 2 * relative_y, w + 2 * relative_y, img.shape[2]), dtype = img.dtype
    )
    
    start_x = max(0, relative_x - x)
    start_y = max(0, relative_y - y)
    extracted_face[
        start_y: start_y + cropped_region.shape[0], start_x: start_x + cropped_region.shape[1]
    ] = cropped_region
    
    return extracted_face, relative_x, relative_y

def align_img_wrt_eyes(
    img: np.ndarray,
    left_eye: Optional[Union[list, tuple]],
    right_eye: Optional[Union[list, tuple]]
) -> Tuple[np.ndarray, float]:
    if left_eye is None or right_eye is None:
        return img, 0

    if img.shape[0] == 0 or img.shape[1] == 0:
        return img, 0
    
    angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))
    
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    
    return img, angle

def project_facial_area(
    facial_area: Tuple[int, int, int, int],
    angle: float,
    size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    direction = 1 if angle >= 0 else -1
    angle = abs(angle) % 30
    if angle == 0:
        return facial_area
    
    angle = angle * np.pi / 180
    
    height, weight = size

    x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
    y = (facial_area[1] + facial_area[3]) / 2 - height / 2

    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    x_new = x_new + weight / 2
    y_new = y_new + height /2

    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), weight)
    y2 = min(int(y2), height)

    return (x1, y1, x2, y2)