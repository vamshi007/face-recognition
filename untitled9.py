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


path01='/Users/vamshibukya/Desktop/a.jpeg'
path02='/Users/vamshibukya/Desktop/b.jpg'


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



extract01=extract_face(path01)
extract02=extract_face(path02)

#embedded01=get_model_score(extract01)
#embedded02=get_model_score(extract02)
a = np.array(extract01)
b = a.ravel()


c = np.array(extract02)
d= c.ravel()



distance=cosine(b,d)
print(distance)
























