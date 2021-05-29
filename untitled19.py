import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

dataset=pd.read_csv('selfmade_csv.csv')


x=dataset[['Score']]

y=dataset[['Decision']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()

#from sklearn.neighbors import KNeighborsClassifier

#model1 = KNeighborsClassifier(n_neighbors=1)

model1.fit(x_train, y_train)

pred_prob1 = model1.predict_proba(x_test)

from sklearn.metrics import roc_curve

fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)

print(thresh1)

from sklearn.metrics import roc_auc_score

auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])


# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()




from sklearn.metrics import confusion_matrix
y_predict_th3 = np.where(model1.predict_proba(x_test)[:,1]>0.88,1,0)
print(confusion_matrix(y_test,y_predict_th3))


from sklearn.metrics import confusion_matrix
y_predict_th4 = np.where(model1.predict_proba(x_test)[:,1]>0.87,1,0)
print(confusion_matrix(y_test,y_predict_th4))

from sklearn.metrics import confusion_matrix
y_predict_th5 = np.where(model1.predict_proba(x_test)[:,1]>0.24,1,0)
print(confusion_matrix(y_test,y_predict_th5))

from sklearn.metrics import confusion_matrix
y_predict_th6 = np.where(model1.predict_proba(x_test)[:,1]>0.90,1,0)
print(confusion_matrix(y_test,y_predict_th6))






























