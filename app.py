from flask import Flask, render_template, request, jsonify
import keras
import numpy as np
import base64
import io

from PIL import Image

import json
from json import JSONEncoder


app = Flask(__name__)

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
model = keras.models.load_model('src/UNET.h5')

#Testing purposes only
image = keras.preprocessing.image.load_img('src/test_img2.png', target_size=(512, 512))
image_matrix = np.expand_dims(image, 2)
image_matrix = image_matrix.reshape((512,512,3))
X = np.empty((1, *(512,512) + (3,)))
X[0,] = image_matrix
prediction = model.predict(X, batch_size=1)
prediction_matrix = combineMask(prediction[0])
image = keras.preprocessing.image.array_to_img(
    prediction_matrix, data_format=None, scale=True, dtype=None,
    )
data = io.BytesIO()
image.save(data, "JPEG")
encoded_img_data = base64.b64encode(data.getvalue())
#End Testing

@app.route('/')
def index():

    return render_template('index.html',
                            username='Syca',
                            prediction=prediction,
                            image_data = encoded_img_data.decode('utf-8')
                            )

@app.route("/predict", methods=["POST"])
def process_image():
    file = request.files['image']
    img = Image.open(file.stream)

    img = preprocessPredictImg(img)

    json_string = json.dumps(img, cls=NumpyArrayEncoder)

    return jsonify({'msg': 'success', 'data': json_string})



if __name__ == "__main__":
    app.run()             
