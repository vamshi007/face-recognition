#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:00:50 2021

@author: vamshibukya
"""
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

dataset=pd.read_csv('/Users/vamshibukya/Desktop/all other stuff/new_selfmade_csv_without.csv')
dataset['Flag'] = 1

x=dataset[['Score']]

y=dataset[['Decision']]

z=dataset[['Flag']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression


model=LogisticRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))


print(model.predict_proba(x_test)[:,1])

#y_predict_th4 = np.where(model.predict_proba(x_test)[:,1]>0.4,1,0)

from sklearn.metrics import roc_curve, roc_auc_score

tpr,fpr,thresholds = roc_curve(y_train,model.predict_proba(x_train)[:,1])

#thresholds






fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
thresholds
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

plt.plot(fpr, tpr,color='red',lw=5)
plt.show()





from sklearn.metrics import confusion_matrix,roc_auc_score

count=0
while(count<=1):
    y_predict_th3 = np.where(model.predict_proba(x_test)[:,1]>count,1,0)
    print(confusion_matrix(y_test,y_predict_th3))
    print(count)
    count=count+0.1
    
roc_auc_score(y_test,model.predict_proba(x_test)[:,1])



from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))



## plot based on density

dataset[dataset.Decision == 1].Score.plot.kde()
dataset[dataset.Decision == 0].Score.plot.kde()














