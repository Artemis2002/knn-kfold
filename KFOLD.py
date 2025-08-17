import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

heart = pd.read_csv('heart.csv')
print(heart.columns)

x = heart [['age' , 'sex' , 'cp' , 'trtbps' , 'chol' , 'fbs' , 'restecg' , 'thalachh' , 'exng' 
            , 'oldpeak' , 'slp' , 'caa' , 'thall']].values.astype(float)
y = heart['output'].values.astype(float)

x_train , x_test , y_train , y_test = train_test_split(x, y )

knn = KNeighborsClassifier().fit(x_train,y_train)
yhat = knn.predict(x_test)

print('-----------------------------yhat----------------------------\n')
print(yhat[0:30])
print('-----------------------------ytest----------------------------\n')
print(y_test[0:30])

   