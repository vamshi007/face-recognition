#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:11:55 2021

@author: vamshibukya
"""
import cv2
import glob
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import os
import imutils
import dlib
import numpy as np

source='/Users/vamshibukya/Desktop/lfw_funneled/'
destination='/Users/vamshibukya/Desktop/synthetic_lfw/'

li=os.listdir(source)
li.remove('.DS_Store')
li_paths=[]
count=0

for i in range(0,len(li)):
    filename=destination+li[i]
    if not os.path.exists(filename):
        os.mkdir(li[i])

for i in range(0,len(li)):
    for j in glob.glob(source+li[i]+'/*.*'):
        li_paths.append(j)
 
        
p = "/Users/vamshibukya/Desktop/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
#synthetic mask
def draw_faces(image_path):
    img=cv2.imread(image_path)
    a=image_path.split('/')
    img=imutils.resize(img,width=500)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    predictor = dlib.shape_predictor(p)
    for face in faces:
        landmarks = predictor(gray, face)
    points = []
    for i in range(1, 16):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)
    mask_c =[((landmarks.part(29).x), (landmarks.part(29).y))]
    fmask_c = points + mask_c
    fmask_c = np.array(fmask_c, dtype=np.int32)
    img2 = cv2.polylines(img, [fmask_c], True,color=(0, 0, 0), thickness=2, lineType=cv2.LINE_8)
    img3 = cv2.fillPoly(img2, [fmask_c],color=(0, 0, 0), lineType=cv2.LINE_AA)
    # plt.imshow(img3)
    plt.savefig(destination+a[-2]+'/'+a[-1])
                
    


for i in range(0,len(li_paths)):
    count=count+1 
    print(count)
    draw_faces(li_paths[i])
    

    



  
    
        




