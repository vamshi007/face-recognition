#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:02:59 2021

@author: vamshibukya
"""
import pandas as pd
import pickle
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from mtcnn import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray


names=pickle.load(open('/Users/vamshibukya/Desktop/verification/selfmade_test_names_with.pickle','rb'))
data =pickle.load(open('/Users/vamshibukya/Desktop/verification/selfmade_test_embedded_with.pickle','rb'))

#names=pd.read_csv('selfmade.csv')
dataframe=pd.DataFrame(np.concatenate(data))

model=VGGFace(model='resnet50')

def extract_face(image_path):
    pixels=plt.imread(image_path)
    image=Image.fromarray(pixels)
    image=image.resize((224,224))
    face_array=asarray(image)
    return face_array


#face Embedded vector for the given extracted face
def get_model_score(img_faces):
    x=img_faces.astype('float64')
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1) 
    return model.predict(x)




test_image='/Users/vamshibukya/Desktop/verification/test_images/Vanessa Kirby.jpg'
img_faces=extract_face(test_image)
model_score_img1=get_model_score(img_faces)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(names)


from sklearn.svm import SVC
clf = SVC(kernel='rbf') 
clf.fit(dataframe, labels) 

index_value=clf.predict(model_score_img1)


index_values=np.where(labels==index_value)

print(index_values)

for i in range(0,len(index_values[0])):
    print(names[index_values[0][i]])
























