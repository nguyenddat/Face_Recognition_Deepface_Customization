from typing import *

from pydantic import BaseModel

class RecognitionRequest(BaseModel):
    img_path: str
    detector_backend: str = "opencv"
    distance_metric: str = "cosine"
    threshold: Optional[float] = None
    
    class Config:
        from_attributes = True
    
    