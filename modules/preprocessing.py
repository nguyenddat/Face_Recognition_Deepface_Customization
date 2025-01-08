from typing import *

import numpy as np
import cv2

from helpers import package_helpers

tf_major_version = package_helpers.get_tf_major_version()
if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image
    
def normalize_input(
    img: np.ndarray,
    normalization: str = "base"
) -> np.ndarray:
    if normalization == "base":
        return img

def resize_image(
    img: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
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