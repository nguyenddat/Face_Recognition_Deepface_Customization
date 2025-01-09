from typing import *

from models.face_detection import OpenCv, RetinaFace
from models.face_recognition import VGGFace

models = {
    "facial_recognition": {
        "VGG-Face": VGGFace.VGGFaceClient,
    },
    "face_detector": {
        "opencv": OpenCv.OpenCvClient,
        "retinaface": RetinaFace.RetinaFaceClient
    }
}

def build_model(task: str, model_name: str) -> Any:
    global cached_models
    
    if not "cached_models" in globals():
        cached_models = {current_task: {} for current_task in models.keys()}
    
    if cached_models[task].get(model_name) is None:
        model = models[task].get(model_name)
        print(f"Found model: {model}")
        if model:
            cached_models[task][model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {task}/{model_name}")
    return cached_models[task][model_name]    