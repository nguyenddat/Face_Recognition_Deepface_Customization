FROM python:3.12.6

RUN apt-get update && apt-get install -y 
RUN apt-get install -y gcc g++ python3-dev musl-dev libffi-dev

RUN mkdir FaceRecognition 
RUN cd FaceRecognition
COPY .env /FaceRecognition

RUN mkdir api
COPY ./api /api

RUN mkdir face_recogntion
COPY ./face_recogntion face_recogntion

WORKDIR /api


