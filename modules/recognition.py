import os
from typing import *

import numpy as np

from helpers import image_helpers

def find(
    img_path: Union[str, np.ndarray],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
    refresh_database: bool = True
):
    if not os.path.isdir(db_path):
        raise ValueError("db path is not existed")
    
    