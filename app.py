import streamlit as st

import keras
import numpy as np

from PIL import Image

import json
from json import JSONEncoder


# Header
st.header("Image Segmentation")
@st.cache

def combineMask(masks):
    _output = np.empty((512,512) + (1,))
    
    for x in range(0, masks.shape[0]):
        for y in range(0, masks.shape[1]):
                   _target = masks[x][y]
                   _output[x][y] = np.argmax(_target) 
    return _output

def preprocessPredictImg(img):
    img = img.resize((512,512))
    image_matrix = np.expand_dims(img, 2)
    image_matrix = image_matrix.reshape((512,512,3))

    X = np.empty((1, *(512,512) + (3,)))
    X[0,] = image_matrix

    prediction = model.predict(X, batch_size=1)
    prediction_matrix = combineMask(prediction[0])

    return prediction_matrix

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

#IMPORTANT : LOAD MODEL (current with img_size=(512,512))
model = keras.models.load_model('UNET.h5')


