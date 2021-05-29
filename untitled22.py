#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:28:08 2021

@author: vamshibukya
"""
import cv2
import glob
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from PIL import Image
import os

source='/Users/vamshibukya/Desktop/selfmade dataset/'
destination='/Users/vamshibukya/Desktop/selfmade/'


li=os.listdir(source)
li.remove('.DS_Store')
li_paths=[]


for i in range(0,len(li)):
    filename = destination + li[i]
    if not os.path.exists(filename):
         os.mkdir(filename)
         os.chdir(filename)
         os.mkdir("with_mask")
         os.mkdir('without_mask')
         
         
mask=['with_mask','without_mask']  

for i in range(0,len(li)):
    for k in range(0,len(mask)):
        for j in glob.glob(source+li[i]+'/'+mask[k]+'/*.*'):
                li_paths.append(j)
                
'''                
count=0               
for i in range(0,len(li)):
    for j in glob.glob(source+li[i]+'/'+mask[1]+'/*.*'):
        count=count+1
        
print(count)'''

detector=MTCNN()


def draw_faces(image_path):
    data=plt.imread(image_path)
    a=image_path.split('/')
    print(a)
    faces=detector.detect_faces(data)
    if 0< len(faces) <=2 :
        try:
            for i in range(len(faces)):
                x1,y1,width,height=faces[i]['box']
                x2,y2=x1+width,y1+height
                plt.subplot(1,len(faces),i+1)
                plt.axis('off')
                plt.imshow(data[y1:y2,x1:x2]) 
                print(destination+a[-2]+'/'+a[-1])
                plt.savefig(destination+a[-3]+'/'+a[-2]+'/'+a[-1],bbox_inches='tight',pad_inches = 0)
            plt.show()
        except ValueError:
            pass
    
    
    
for i in range(2981,len(li_paths)):
    print(i)
    draw_faces(li_paths[i])