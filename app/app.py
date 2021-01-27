from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import Flask
from markupsafe import escape
from flask import request
from flask import make_response
from flask_cors import CORS
from flask.json import jsonify
from binascii import a2b_base64
import base64
import urllib.request
import io
import time

import numpy as np
from PIL import Image
# import tensorflow as tf # TF2
import tflite_runtime.interpreter as tflite


app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/analysis', methods=['POST'])
def analysis():
    load_labels()
    if request.data:
        imagetostore = request.data[request.data.find(b'/9'):]
        image = Image.open(io.BytesIO(base64.b64decode(imagetostore))).save('image.jpg')
        print('file detected')
        interpreter = tflite.Interpreter(
        model_path='modele_cree6.tflite', num_threads=None)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # check the type of the input tensor
        floating_model = input_details[0]['dtype'] == np.float32

        # NxHxWxC, H:1, W:2
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        img = Image.open('image.jpg').resize((width, height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - 0) / 255

        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels()

        response_list = []

        for i in top_k:
            if floating_model:
                print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
                response_list.append({"value": float(results[i]), "species": labels[i]})
            else:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
                response_list.append({"value": float(results[i] / 255.0), "species": labels[i]})

        print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
        print(response_list)
        return jsonify(response_list)
    return {"message": "no file"}

# Loads species labels
def load_labels():
  with open('class_labels6.txt', 'r') as f:
    return [line.strip() for line in f.readlines()]

def run()
    app.run(debug=True)
    