from typing import *

from pydantic import BaseModel

class RecognitionRequest(BaseModel):
    data: str
    model_name: str = "VGG-Face"
    detector_backend: str = "opencv"
    distance_metric: str = "cosine"
    threshold: Optional[float] = None
    
    class Config:
        from_attributes = True
    
    