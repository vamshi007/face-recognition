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
import time
from scipy.spatial.distance import cosine

face_names=pickle.load(open('/Users/vamshibukya/Desktop/verification/selfmade_test_names_with.pickle','rb'))
face_embedded=pickle.load(open('/Users/vamshibukya/Desktop/verification/selfmade_test_embedded_with.pickle','rb'))



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


import os
li=os.listdir()
li.remove('.DS_Store')


abc=0
for i in range(len(li)):
    score_store=[]
    face_store=[]
    count=0
    print('-------------------------')
    print(li[i])
    test_image=li[i]
    equal_01=test_image.split('.')[0]
    img_faces=extract_face(test_image)
    model_score_img1=get_model_score(img_faces)
    for i in range(len(face_embedded)):
        distance=cosine(model_score_img1,face_embedded[i])
        count=count+1
        if (distance <0.80):
            score_store.append(distance)
            face_store.append(face_names[count-1])
    try:
        s=min(score_store)
        print(s)
        index=score_store.index(s)
        show=face_store[index]
        print(show)
    except :
        print('face not found')
        abc=abc+1
print(abc)
               
   
    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                          


    
    
    
    