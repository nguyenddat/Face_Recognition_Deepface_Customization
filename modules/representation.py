from typing import *

import numpy as np
from heapq import nlargest

from  helpers import image_helpers
from modules import modeling, detection, preprocessing
from schemas.FacialRecognition import FacialRecogition

def represent(img: np.ndarray,
              model_name: str = "VGG-Face",
              detector_backend: str = "opencv",
              align: bool = True):
    model: FacialRecogition = modeling.build_model(task = "facial_recognition",
                                                   model_name = model_name)
    target_size =  model.input_shape
    
    img = img[:, :, ::-1]
    img = preprocessing.resize_image(img = img,
                                     target_size = (target_size[1], target_size[0]))
    embedding = model.forward(img)
    return embedding