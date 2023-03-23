# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 06:17:03 2021

@author: Ahmed Fayed
"""

# from keras_facenet import FaceNet
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.core.numeric import identity
from sklearn.preprocessing import Normalizer
from PIL import Image
from deepface.basemodels import VGGFace, Facenet
from flask import Flask, render_template, request, send_from_directory, url_for, redirect, jsonify, json
import codecs
import base64
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst
from deepface import DeepFace

embedder = Facenet.loadModel()
l2_normalizer = Normalizer('l2')


app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

#HOME
@app.route('/')
def home():
    return render_template('options.html')


#REPRESENT
@app.route('/represent', methods = ['POST'])
def represent():
    return render_template('choosemodels_represent.html')

@app.route('/represent/facenet', methods = ['POST'])
def facenet():
    return render_template('represent_facenet.html')

@app.route('/represent/vggface', methods = ['POST'])
def vggface():
    return render_template('represent_vggface.html')
    
#verify
@app.route('/verify', methods = ['POST'])
def verify():
    return render_template('choosemodels_verify.html')

@app.route('/verify/facenet', methods = ['POST'])
def facenet1():
    return render_template('verify_facenet.html')

@app.route('/verify/vggface', methods = ['POST'])
def vggface1():
    return render_template('verify_vggface.html')


#analyze
@app.route('/analyze', methods = ['POST'])
def analyze():
    return render_template('analyze.html')

#-----------------------------------------------------------------------------------------------------------------

# PRESENT 
@app.route('/represent/vggface/encoding', methods=['POST'])
def encoding():
    model = VGGFace.loadModel()
    img_upload = request.files['image']
    img = Image.open(img_upload) #Mở ảnh
    img_arr = np.array(img) #Chuyển ảnh về numpy_array 
    # cv2.imwrite("anh1.png", img_arr)  
    # img1 = cv2.imread('anh1.png')
    
    input_shape_x, input_shape_y = functions.find_input_shape(model)
    img3 = functions.preprocess_face(img = img_arr, target_size=(input_shape_y, input_shape_x), enforce_detection = True, detector_backend = 'opencv', align = True)
 
    img_nor = functions.normalize_input(img = img3, normalization = 'base')
    embedding_list = model.predict(img_nor)[0].tolist()
    return jsonify({"embedding" : embedding_list})

@app.route('/represent/facenet/encoding', methods=['POST'])
def encoding1():
    model = Facenet.loadModel()
    img_upload = request.files['image']
    img = Image.open(img_upload) #Mở ảnh
    img_arr = np.array(img) #Chuyển ảnh về numpy_array 
    # cv2.imwrite("anh1.png", img_arr)  
    # img1 = cv2.imread('anh1.png')
    
    input_shape_x, input_shape_y = functions.find_input_shape(model)
    img3 = functions.preprocess_face(img = img_arr, target_size=(input_shape_y, input_shape_x), enforce_detection = True, detector_backend = 'opencv', align = True)
 
    img_nor = functions.normalize_input(img = img3, normalization = 'base')
    embedding_list = model.predict(img_nor)[0].tolist()
    return jsonify({"embedding" : embedding_list})
 
#-----------------------------------------------------------------------------------------------------------------
#VERIFY
@app.route('/verify/vggface/distance', methods=['POST'])
def distance():
    embedder = VGGFace.loadModel()
    l2_normalizer = Normalizer('l2')
    

    img_upload = request.files['image']
    img2_upload = request.files['image2']
    

    img = Image.open(img_upload)
    img_arr = np.array(img)

    img2 = Image.open(img2_upload)
    img2_arr = np.array(img2)

    #Anh 1 
    input_shape_x, input_shape_y = functions.find_input_shape(embedder)
    img3 = functions.preprocess_face(img = img_arr, target_size=(input_shape_y, input_shape_x), enforce_detection = True, detector_backend = 'opencv', align = True)
 
    img_nor = functions.normalize_input(img = img3, normalization = 'base')
    embedding1 = embedder.predict(img_nor)[0].tolist()
    
    #Anh 2
    input_shape_x, input_shape_y = functions.find_input_shape(embedder)
    img4 = functions.preprocess_face(img = img2_arr, target_size=(input_shape_y, input_shape_x), enforce_detection = True, detector_backend = 'opencv', align = True)
 
    img_nor = functions.normalize_input(img = img4, normalization = 'base')
    embedding2 = embedder.predict(img_nor)[0].tolist()

    # distance = embedder.compute_distance(embedding1, embedding2)
    distance = dst.findEuclideanDistance(dst.l2_normalize(embedding1), dst.l2_normalize(embedding2))
    distance = np.float64(distance) 
    distance = np.round(distance, decimals=1)
            

    #Đặt ngưỡng đúng sai 
    # threshold = dst.findThreshold(embedder, l2_normalizer)
    threshold = 0.8

    if distance <= threshold:
        identity = True #2 ảnh là cùng một người 
    else:
        identity = False #2 ảnh không cùng một người

    return jsonify({"identity":identity})

@app.route('/verify/facenet/distance', methods=['POST'])
def distance1():
    embedder = Facenet.loadModel()
    l2_normalizer = Normalizer('l2')
    

    img_upload = request.files['image']
    img2_upload = request.files['image2']
    

    img = Image.open(img_upload)
    img_arr = np.array(img)

    img2 = Image.open(img2_upload)
    img2_arr = np.array(img2)

    #Anh 1 
    input_shape_x, input_shape_y = functions.find_input_shape(embedder)
    img3 = functions.preprocess_face(img = img_arr, target_size=(input_shape_y, input_shape_x), enforce_detection = True, detector_backend = 'opencv', align = True)
 
    img_nor = functions.normalize_input(img = img3, normalization = 'base')
    embedding1 = embedder.predict(img_nor)[0].tolist()
    
    #Anh 2
    input_shape_x, input_shape_y = functions.find_input_shape(embedder)
    img4 = functions.preprocess_face(img = img2_arr, target_size=(input_shape_y, input_shape_x), enforce_detection = True, detector_backend = 'opencv', align = True)
 
    img_nor = functions.normalize_input(img = img4, normalization = 'base')
    embedding2 = embedder.predict(img_nor)[0].tolist()

    # distance = embedder.compute_distance(embedding1, embedding2)
    distance = dst.findEuclideanDistance(dst.l2_normalize(embedding1), dst.l2_normalize(embedding2))
    distance = np.float64(distance) 
    distance = np.round(distance, decimals=1)
            

    #Đặt ngưỡng đúng sai 
    # threshold = dst.findThreshold(embedder, l2_normalizer)
    threshold = 0.5

    if distance <= threshold:
        identity = True #2 ảnh là cùng một người 
    else:
        identity = False #2 ảnh không cùng một người

    return jsonify({"identity":identity})
#-------------------------------------------------------------------------------------------------------------------

#ANALYZE
@app.route('/analyzee', methods=['POST'])
def analyzee():
    age = DeepFace.build_model('Age')
    gender = DeepFace.build_model('Gender')
    emotion = DeepFace.build_model('Emotion')
    race = DeepFace.build_model('Race')
    
    img_upload = request.files['image']
    img = Image.open(img_upload) #Mở ảnh
    img_arr = np.array(img) #Chuyển ảnh về numpy_array 
    
    
    obj = DeepFace.analyze(img_arr, actions = ['age', 'gender', 'race', 'emotion'])
    
    return jsonify({"obj" : obj})












if __name__ == '__main__':
    app.run(debug=True)

