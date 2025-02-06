from fastapi import APIRouter, HTTPException, status

from ..schemas.base import DataResponse
from ..schemas.recognition import RecognitionRequest
from ..services.FaceRecognition import face_recognition

router = APIRouter()

@router.post("/api/recognition")
def recognize(api_data: RecognitionRequest):
    if not api_data:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail = "Data is not found")

    print(api_data.detector_backend)
    try:
        resp = face_recognition.find(
            img_path = api_data.img_path,
            detector_backend = api_data.detector_backend,
            distance_metric = api_data.distance_metric,
            threshold = api_data.threshold
        )
        
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail = err) from err
        
    return resp    
    
