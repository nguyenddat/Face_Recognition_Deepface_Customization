FROM python:3.12.6

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    musl-dev \
    libffi-dev \
    libgl1-mesa-glx

RUN mkdir FaceRecognition 
COPY . FaceRecognition
WORKDIR /FaceRecognition

RUN pip install -r requirements.txt
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]