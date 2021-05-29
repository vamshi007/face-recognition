#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:37:30 2021

@author: vamshibukya
"""
from numpy import expand_dims
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from numpy import asarray
from PIL import Image
from matplotlib.patches import Rectangle


from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
import numpy as np
from keras_vggface import utils


faces=[]
detector=MTCNN()

def print_faces(image_path):
    image=plt.imread(image_path)
    detector=MTCNN()
    faces=detector.detect_faces(image)
    for face in faces:
        print(face)
        print("number of faces detected",len(faces))
    highlight_face(image_path,faces)
    
def highlight_face(filename, result_list):
	data = plt.imread(filename)
	plt.imshow(data)
	ax = plt.gca()
	for result in result_list:
		x, y, width, height = result['box']
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		ax.add_patch(rect)
	plt.show()
    
def draw_faces(image_path,faces):
    data=plt.imread(image_path)
    faces=detector.detect_faces(data)
    for i in range(len(faces)):
        x1,y1,width,height=faces[i]['box']
        x2,y2=x1+width,y1+height
        plt.subplot(1,len(faces),i+1)
        plt.axis('off')
        plt.imshow(data[y1:y2,x1:x2])
    plt.show()
    
    
def extract_face(image_path):
    pixels=plt.imread(image_path)
    detector=MTCNN()
    results=detector.detect_faces(pixels)
    x1,y1,width,height=results[0]['box']
    x2,y2=x1+width,y1+height
    face=pixels[y1:y2,x1:x2]
    image=Image.fromarray(face)
    image=image.resize((224,224))
    face_array=asarray(image)
    return face_array
    
    
    
def model_pred(faces):
    samples=faces.astype('float32')
    samples=expand_dims(samples,axis=0)
    samples=preprocess_input(samples,version=2)
    model=VGGFace(model='senet50')
    print('Input: %s' %model.inputs)
    print('Output: %s'%model.outputs)
    yhat=model.predict(samples)
    return yhat

def decoder(yhat):
    results=decode_predictions(yhat)
    for result in results[0]:
        print('the likely candidate is %s with confidance level of %.3f%%'%(result[0],result[1]*100))
    
'''   

def get_model_score(faces):
    img = image.load_img(faces, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1) # or version=2
    model=VGGFace(model='vgg16')
    preds = model.predict(x)
    print('Predicted:', utils.decode_predictions(preds))

    

    
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    