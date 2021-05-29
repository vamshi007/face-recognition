import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from PIL import Image


path='/Users/vamshibukya/Desktop/selfmade/'
li=os.listdir(path)
li.remove('.DS_Store')
mask=['with_mask','without_mask']

count=100
counter=0
for i in range(0,len(li)):
    print(counter)
    counter=counter+1
    for j in range(0,len(mask)):
        for img in glob.glob(path+li[i]+"/"+mask[j]+"/*.*"):
            imgs=Image.open(img)
            horizontal_flip=np.fliplr(imgs)
            data = Image.fromarray(horizontal_flip)
            path01=path+li[i]+"/"+mask[j]
            data.save(path01+"/"+str(count)+'.jpg')
            count=count+1
            
                
            
            
            
        
            

