from fastapi import APIRouter, HTTPException, status

from ..core import config
from ..schemas.base import DataResponse
from ..schemas.recognition import RecognitionRequest

from face_recognition.modules import recognition

router = APIRouter()

@router.post("/api/recognition")
def recognize(api_data: RecognitionRequest):
    if not api_data:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail = "Data is not found")

    try:
        resp = recognition(
            img_path = api_data.data,
            db_path = str(config.settings.DB_PATH),
            model_name = api_data.model_name,
            detector_backend = api_data.detector_backend,
            distance_metric = api_data.distance_metric,
            align = api_data.align,
            threshold = api_data.threshold,
            normalize_face = api_data.normalize_face
        )

    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail = err) from err
        
    return DataResponse().success_response(data = resp)
    
    
