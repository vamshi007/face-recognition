#importing related lib
import os
import numpy as np
import glob

from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import numpy as np
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
import cv2
import pickle
import pandas as pd
import sklearn


#Extracting face from image and converting into an array
def extract_face(image_path):
    pixels=cv2.imread(image_path)
    image=Image.fromarray(pixels)
    image=image.resize((224,224))
    face_array=asarray(image)
    return face_array

model=VGGFace(model='resnet50')
#face Embedded vector for the given extracted face
def get_model_score(img_faces):
    x=img_faces.astype('float64')
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1) 
   
    return model.predict(x)

test_image='/Users/vamshibukya/Desktop/selfmade/prince william/without_mask/8.jpg'
test_image01='/Users/vamshibukya/Desktop/selfmade/prince william/without_mask/10.jpg'

img_faces=extract_face(test_image)
model_score_img1=get_model_score(img_faces)

img_faces01=extract_face(test_image01)
model_score_img01=get_model_score(img_faces01)


similarity=cosine(model_score_img1,model_score_img01)

print(similarity)






















