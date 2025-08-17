import sklearn as sk
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
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

kf = KFold(n_splits=5, shuffle=True, random_state=32)

m_scores = np.array([])

for j in range(2,25):
    print(f"----------------------------------------{j}--------------------------------------")
    knn = KNeighborsClassifier(n_neighbors=j)
    scores = cross_val_score(knn, x, y, cv=kf, scoring='accuracy')
    print(f"Accuracy scores for each fold: {scores}")
    print(f"Mean accuracy for K={j}: {scores.mean()}\n\n")
    m_scores = np.append(m_scores , scores.mean())
 
print(f'best k is=' , np.argmax(m_scores) + 2)   
print(f'Best mean accuracy is=' , np.max(m_scores))


