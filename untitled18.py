import os
import shutil

target_dir='/Users/vamshibukya/Desktop/Real-World-Masked-Face-Dataset-master/h/'
source_dir='/Users/vamshibukya/Desktop/Real-World-Masked-Face-Dataset-master/RWMFD_part_2_pro'


count=0
for path,dir,files in os.walk(source_dir):
    for f in files:
        if '.DS_Store' in f:
            print("ds file was here")
        else:
            shutil.move(path+'/'+f,target_dir)
        for path,dir,files in os.walk(target_dir):
            for f in files:
                os.rename(f,str(count)+".jpg")
                int(count=count+1)