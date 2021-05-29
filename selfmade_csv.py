import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import pickle
import time
face_embedded=pickle.load(open('selfmade_embedded.pickle','rb'))
face_names=pickle.load(open('selfmade_names.pickle','rb'))



df=pd.DataFrame(columns=['Name01', 'Name02','Score','Decision'])

_start = time.time()
for i in range(0,50):
    start = time.time()
    print(len(face_names))
    for j in range(0,len(face_names)):
        #print(face_embedded[i].shape,face_embedded[j].shape)
        cosine_distance=cosine(face_embedded[i],face_embedded[j])
        yes_or_no = "Yes" if face_names[i]==face_names[j] else "No"
        df1=[{"Name01":face_names[i],
                         "Name02":face_names[j],
                         "Score":cosine_distance,
                         "Decision":yes_or_no}]
        b = pd.DataFrame.from_dict(df1)
        df = df.append(b,ignore_index=True)
    end = time.time()
    print("Time taken for {}: {}".format(i, (end-start)))
df.to_csv('selfmade_csv.csv')
_end = time.time()
print("Time taken for {}: {}".format(i, (_end-_start)))