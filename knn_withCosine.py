#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:07:31 2021

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
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score

names=pickle.load(open('/Users/vamshibukya/Desktop/revised/selfmade_all_revised_test_names_with.pickle','rb'))
data =pickle.load(open('/Users/vamshibukya/Desktop/revised/selfmade_all_revised_test_embedded_with.pickle','rb'))

def cosine_distance(x,y):
    return cosine(x,y)

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






dataframe=pd.DataFrame(np.concatenate(data))
load_name=[]
for i in range(0,len(names)):
    name=names[i].split('/')
    load_name.append(name)
    


from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=9, metric=cosine_distance)


le = LabelEncoder()
labels = le.fit_transform(load_name)

neigh.fit(dataframe,labels)


import os
import time

test_data=os.listdir()
test_data.remove('.DS_Store')

for i in range(len(test_data)):
    
    print(test_data[i])
    test_image='/Users/vamshibukya/Desktop/verification/test_images/'+test_data[i]
    img_faces=extract_face(test_image)
    model_score_img1=get_model_score(img_faces)
    
    
    
    index_value=neigh.predict(model_score_img1)
    
    index_values=np.where(labels==index_value)
    
    #print(index_values)
    
    
    for i in range(0,len(index_values[0])):
        compare=names[index_values[0][i]]
    time.sleep(3)
    
    print(compare)
    print('----------------------------')
    
    













