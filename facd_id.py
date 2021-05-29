#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:49:50 2021

@author: vamshibukya
"""
import os
import glob

from face_in import *


test_image_path='image/Obama_masked5.jpg'

print("print face and draw a bounding box")

faces=print_faces(test_image_path)
draw_faces(test_image_path,faces)

print('___________________________')


print('face Identification with VGGFace2')
pixels=extract_face(test_image_path)

print('predicting name from image')

decoder(model_pred(pixels))