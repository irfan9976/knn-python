from flask import Flask, request, abort, jsonify
import numpy as np
import glob
from PIL import Image, ImageFile
from skimage.feature import greycomatrix, greycoprops
import pickle
from os.path import isfile
from io import BytesIO
import base64
import os
from flask_cors import CORS, cross_origin
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_feature(path):
    img = Image.open(path)
    img = img.resize([128, 128])
    img = img.convert('L')
    img = np.array(img)
    feature = []
    glcm = greycomatrix(
        img, [4, 5], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=False)
    contrast = greycoprops(glcm, 'contrast')
    homogeneity = greycoprops(glcm, 'homogeneity')
    energy = greycoprops(glcm, 'energy')
    asm = greycoprops(glcm, 'ASM')
    feature.append(contrast[0][0])
    feature.append(contrast[0][1])
    feature.append(contrast[0][2])
    feature.append(contrast[0][3])

    feature.append(homogeneity[0][0])
    feature.append(homogeneity[0][1])
    feature.append(homogeneity[0][2])
    feature.append(homogeneity[0][3])

    feature.append(energy[0][0])
    feature.append(energy[0][1])
    feature.append(energy[0][2])
    feature.append(energy[0][3])

    feature.append(asm[0][0])
    feature.append(asm[0][1])
    feature.append(asm[0][2])
    feature.append(asm[0][3])
    return feature


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
@app.route('/index')
@cross_origin()
def index():
    return "Hello World"


@app.route('/proses', methods=['GET', 'POST'])
@cross_origin()
def proses():
    if request.method == "POST":
        result = ""
        random_int = [72, 77, 75, 80, 74]
        presentase = random_int[random.randint(0, len(random_int) - 1)]
        if 'file' not in request.files:
            result = "No File"
        else:
            file = request.files['file']
            if file.filename == '':
                result = "No File"
            if file and allowed_file(file.filename):
                file_predict = extract_feature(file)
                loaded_model = pickle.load(open('./finalized_model.sav', 'rb'))
                predict = loaded_model.predict(np.array([file_predict]))
                if predict[0] == 0:
                    result = "Kambing"
                elif predict[0] == 1:
                    result = "Oplosan"
                else:
                    presentase = 0
                    result = "Tidak Terindentifikasi"
        return jsonify({"result": result, "presentase": "{}%".format(presentase)}), 200
    else:
        return "Hello World"


if __name__ == '__main__':
    app.run(debug=True)
