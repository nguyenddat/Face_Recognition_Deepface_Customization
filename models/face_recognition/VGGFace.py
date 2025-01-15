from typing import *

import numpy as  np

from helpers import package_helpers, weight_helpers, model_helpers
from helpers.logger import logger 
from schemas.FacialRecognition import FacialRecogition

tf_version = package_helpers.get_tf_major_version()
if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import (Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation)
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation)

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5"
)

class VGGFaceClient(FacialRecogition):
    def __init__(self):
        self.model = self.load_model()
        self.model_name = "VGG-Face"
        self.input_shape = (224, 224)
        self.output_shape = 4096
    
    def forward(self, img: np.ndarray) -> List[float]:
        embedding = self.model(img, training = False).numpy()[0].tolist()
        embedding = model_helpers.l2_normalize(embedding)
        return embedding.tolist()
    
    def load_model(self, url = WEIGHTS_URL) -> Model:
        model = self.base_model()
        weight_file = weight_helpers.download_weights_if_necessary(
            file_name = "vgg_face_weights.h5", source_url=url
        )
        
        model = weight_helpers.load_model_weights(model = model, weight_file = weight_file)

        base_model_output = Flatten()(model.layers[-5].output)
        
        vgg_face_descriptor = Model(inputs = model.inputs, outputs = base_model_output)
        return vgg_face_descriptor

    def base_model(self) -> Sequential:
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(4096, (7, 7), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation("softmax"))

        return model        
        