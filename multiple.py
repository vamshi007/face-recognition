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
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import cv2

face_names=pickle.load(open('/Users/vamshibukya/Desktop/verification/selfmade_test_names_with.pickle','rb'))
face_embedded=pickle.load(open('/Users/vamshibukya/Desktop/verification/selfmade_test_embedded_with.pickle','rb'))

model=VGGFace(model='resnet50')
detector=MTCNN()



#Extracting face from image and converting into an array



def extract_face(image_path):
    pixels=plt.imread(image_path)
    results=detector.detect_faces(pixels)
    for i in range(len(results)):        
        x1,y1,width,height=results[i]['box']
        x2,y2=x1+width,y1+height
        face=pixels[y1:y2,x1:x2]
        image=Image.fromarray(face)
        image=image.resize((224,224))
        face_array=asarray(image)
        x=face_array.astype('float64')
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1) 
        embeddings.append(model.predict(x))



    
 


def draw(test_image):
    count=0
    image=plt.imread(test_image)
    plt.imshow(image)
    faces = detector.detect_faces(image)
    for result in faces:
         x,y, width, height = result['box']
         ax=plt.gca()
         rect = Rectangle((x, y), width, height, fill=False, color='red')
         ax.add_patch(rect)
         ax.text(x,y,str(draw_score[count])+"\n"+str(draw_name[count]))
         count=count+1
    plt.show()
    






test_image='/Users/vamshibukya/Desktop/d.jpg'



embeddings=[]
draw_score=[]
draw_name=[]
extract_face(test_image)
#draw(test_image)
for j in range(len(embeddings)):
    score_store=[]
    face_store=[]
    count=0
    for i in range(len(face_embedded)):
        distance=cosine(embeddings[j],face_embedded[i])
        count=count+1
        if (distance <0.70):
            score_store.append(distance)
            face_store.append(face_names[count-1])
    try:
        s=min(score_store)
        print(s)
        
        index=score_store.index(s)
        show=face_store[index]
        print(show)
        draw_score.append(s)
        draw_name.append(show)
    except :
        draw_score.append(1)
        draw_name.append('Unknown')
        print('face not found')
draw(test_image)


    


        
    









    
    









