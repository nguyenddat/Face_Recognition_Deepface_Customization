from typing import *

import numpy as np
import cv2

from ..helpers import package_helpers

tf_major_version = package_helpers.get_tf_major_version()
if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image
    
def normalize_input(img: np.ndarray, 
                    normalization: str = "base") -> np.ndarray:
    if normalization == "base":
        return img

def resize_image(img: np.ndarray,
                 target_size: Tuple[int, int]) -> np.ndarray:
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor)
    )
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 =  target_size[1] - img.shape[1]

    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0)
        ),
        "constant"
    )
    
    if img.shape[0: 2] != target_size:
        img = cv2.resize(img, target_size)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    if img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)

    return img

def extract_sub_img(img: np.ndarray,
                      facial_area: Tuple[int, int, int, int]):
    x, y, w, h = facial_area
    
    # giá trị padding ban đầu
    relative_x = int(0.5 * x)
    relative_y = int(0.5 * y)
    
    # tọa độ mới khi mở rộng (cộng padding vào trên-dưới, trái-phải)
    x1 = x - relative_x
    y1 = y - relative_y
    x2 = x + w + relative_x
    y2 = y + h + relative_y

    # nếu mọi tọa độ đều nằm trong ảnh, trực tiếp trả về khung ảnh mới với tọa độ mới
    if (x1 >= 0) and (y1 >= 0) and (x2 <= img.shape[1]) and (y2 <= img.shape[0]):
        return img[y1: y2, x1: x2], relative_x, relative_y
    else:
        # ngược lại, tính độ lệnh và ánh xạ các điểm từ ảnh cũ sang 1 ảnh đen mới
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, img.shape[1])
        y2 = min(y2, img.shape[0])
        cropped_region = img[y1: y2, x1: x2]

        black_img = np.zeros(
            (h + 2 * relative_y, w + 2 * relative_x, img.shape[2]), dtype = img.dtype
        )
        
        start_x = max(0, relative_x - x)
        start_y = max(0, relative_y - y)

        black_img[start_y: start_y + cropped_region.shape[0],
                  start_x: start_x + cropped_region.shape[1]] = cropped_region
        
        return black_img, relative_x, relative_y

def align_img_with_eyes(img: np.ndarray,
                        left_eye: Tuple[int, int],
                        right_eye: Tuple[int, int]):
    angle = float(np.degrees(
        np.arctan2(
            left_eye[1] - right_eye[1],
            left_eye[0] - right_eye[0]
        )
    ))
    
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center = center,
                                angle = angle,
                                scale = 1.0)
    img = cv2.warpAffine(src = img,
                         M = M,
                         dsize = (w, h),
                         flags = cv2.INTER_CUBIC,
                         borderMode = cv2.BORDER_CONSTANT,
                         borderValue = (0, 0, 0))
    
    return img, angle

def project_facial_area(facial_area: Tuple[int, int, int, int],
                        angle: float,
                        size: Tuple[int, int]):
    direction = 1 if angle >= 0 else -1
    
    angle = abs(angle) % 360
    if angle == 0:
        return facial_area
    
    angle = angle * np.pi / 180
    height, weight = size
    
    x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
    y = (facial_area[1] + facial_area[3]) / 2 - height / 2
    
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)
    
    x_new = x_new + weight / 2
    y_new = y_new + height / 2
    
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2
    
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), weight)
    y2 = min(int(y2), height)
    
    return (x1, y1, x2, y2)    