from typing import *

from pydantic import BaseModel

class RecognitionRequest(BaseModel):
    data: str
    model_name: str = "VGG-Face"
    detector_backend: str = "opencv"
    distance_metric: str = "cosine"
    align: bool = True
    threshold: Optional[float] = None
    normalize_face: bool = True
    
    class Config:
        from_attributes = True
    
    