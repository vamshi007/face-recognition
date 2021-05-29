import cv2
import dlib
import numpy as np
import os
import imutils
import matplotlib.pyplot as plt
from PIL import Image 

destination_path = "/Users/vamshibukya/Desktop/verification/image/"
source='image/Aaron_Peirsol_0001.jpg'
img= cv2.imread(source)
plt.imshow(img)
img = imutils.resize(img, width = 500)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
faces = detector(gray, 1)

print(faces)
print("Number of faces detected: ", len(faces))

p = "/Users/vamshibukya/Desktop/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)

for face in faces:
    landmarks = predictor(gray, face)
    
points = []
for i in range(1, 16):
    point = [landmarks.part(i).x, landmarks.part(i).y]
    points.append(point)

mask_c =[((landmarks.part(29).x), (landmarks.part(29).y))]

'''[((landmarks.part(35).x), (landmarks.part(35).y)),
              ((landmarks.part(34).x), (landmarks.part(34).y)),
              ((landmarks.part(33).x), (landmarks.part(33).y)),
              ((landmarks.part(32).x), (landmarks.part(32).y)),
              ((landmarks.part(31).x), (landmarks.part(31).y))]'''

fmask_c = points + mask_c

fmask_c = np.array(fmask_c, dtype=np.int32)

img2 = cv2.polylines(img, [fmask_c], True,color=(0, 0, 0), thickness=2, lineType=cv2.LINE_8)
img3 = cv2.fillPoly(img2, [fmask_c],color=(0, 0, 0), lineType=cv2.LINE_AA)
plt.imshow(img3)
abc=cv2.imwrite('Aaron_Peirsol_0001_1.jpg',img3)



