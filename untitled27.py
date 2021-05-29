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

known_path='/Users/vamshibukya/Desktop/lfw/'

face_names=[]
embedding_vector=[]

li=os.listdir(known_path)
li.remove('.DS_Store')
mask=['with_mask','without_mask']



def extract_face(image_path):
    pixels=plt.imread(image_path)
    image=Image.fromarray(pixels)
    image=image.resize((224,224))
    face_array=asarray(image)
    return face_array

model=VGGFace(model='resnet50')

def get_model_score(img_path):
    img_faces=extract_face(img_path)    
    x=img_faces.astype('float64')
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=2) 
    print(li[i])
    return model.predict(x)


for i in range(0,len(li)):
    for j in range(0,len(mask)):
        for img_path in glob.glob(known_path+li[i]+'/'+mask[j]+"/"+"*.*"):
            model_score=get_model_score(img_path)
            face_names.append(li[i])
            embedding_vector.append(model_score)




pickle.dump(face_names,open('lfw_names.pickle','wb'))
pickle.dump(embedding_vector,open("lfw_embedded_vectors.pickle",'wb'))
















