#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:15:42 2021

@author: vamshibukya
"""
import os
import numpy as np
import glob
from face_in import *
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from keras_vggface.vggface import VGGFace
import matplotlib.pyplot as plt
import numpy as np

known_path='database/'
test_image='image/Arnold_Schwarzenegger_013.jpg'

print('detect face and draw a bounding box')
faces=print_faces(test_image)
draw_faces(test_image,faces)

print('_______________________')

thres_cosine=0.90


def get_model_score(img_faces):
    x=img_faces.astype('float64')
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1) 
    model=VGGFace(model='resnet50')
    return model.predict(x)

def compare_face(model_score_img1,model_score_img2):
    for idx,face_score_1 in enumerate(model_score_img1):
        for idy,face_score_2 in enumerate(model_score_img2):
            score=cosine(face_score_1,face_score_2)
            if score<=thres_cosine:
                print("there is a match",score)
            else:
                print("this is No match",score)
                
                





print('face verification with vggface2')
img_faces=extract_face(test_image)
model_score_img1=get_model_score(img_faces)


for img2_path in glob.glob(known_path+'*.*'):
    img2_path=extract_face(img2_path)
    model_score_img2=get_model_score(img2_path)
    plt.imshow(img2_path)
    plt.show()
    print("comparing")
    compare_face(model_score_img1,model_score_img2)
    print("next image")




            