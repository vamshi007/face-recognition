import cv2
import pickle
import os
import math
import glob
from scipy.spatial.distance import cosine

names_path='/Users/vamshibukya/Desktop/inital_selfmade/names.pickle'
hu_inv_path='/Users/vamshibukya/Desktop/inital_selfmade/hu_moment.pickle'

path='/Users/vamshibukya/Desktop/verification/test_images/'

names=pickle.load(open(names_path,'rb'))
data =pickle.load(open(hu_inv_path,'rb'))

test_names=os.listdir()
test_names.remove('.DS_Store')

test_image='Adam_Kinzinger.jpg'

im = cv2.imread(test_image,cv2.IMREAD_GRAYSCALE)
_,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
moments = cv2.moments(im)
huMoments = cv2.HuMoments(moments)
for i in range(0,7):
    huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))

result_name=[]
result_score=[]

for i in range(len(data)):
    distances=cosine(huMoments,data[i])
    if distances<0.01:
        result_name.append(names[i])
        result_score.append(distances)
        
        