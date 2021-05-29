import cv2
import os
import math
import pickle

li_dir=os.listdir()
li_dir.remove('.DS_Store')
mask=['with_mask','without_mask']
path='/Users/vamshibukya/Desktop/inital_selfmade/'

import glob
names=[]
hu_inv_moment=[]

for i in range(len(li_dir)):
    for j in range(len(mask)):
        for img in glob.glob(path+li_dir[i]+'/'+mask[j]+"/"+"*.*"):
            im = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            _,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(im)
            huMoments = cv2.HuMoments(moments)
            for i in range(0,7):
                huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
            names.append(li_dir[i])
            hu_inv_moment.append(huMoments)


pickle.dump(names,open('names.pickle','wb'))
pickle.dump(hu_inv_moment,open("hu_moment.pickle",'wb'))





