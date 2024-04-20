# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:16:03 2020

@author: sadievrenseker
"""


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# logistik regresyon
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)# xtrainden ytraini tahmin et  fit eğitmek demek eğitim var 

y_pred = logr.predict(X_test)# tahmin etme işlemi predict tahmin et demek
print(y_pred)
print(y_test)

#karmaşıklık matrisi 
#Yapay zekada karmaşıklık matrisi, bir modelin veya algoritmanın karmaşıklığını analiz etmek ve değerlendirmek için kullanılan bir araçtır.
#Karmaşıklık matrisleri, bir modelin veya algoritmanın performansını etkileyen farklı faktörlerin görsel bir temsili olarak hizmet eder.
#Bu kod, modelin farklı sınıflar arasında ne kadar doğru tahmin yaptığını, yanlış tahminlerin ne olduğunu ve bu hataların hangi sınıflara ait olduğunu görselleştirmek için bir matris oluşturur.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


# en yakın komşu algoritması önemli olan bunu 3 boyutluya çevirmek
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)



from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)


# naif bayes olasılıksal tahmin
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

# karar ağacı (kötü yöntem)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)
#random forrest , rassal ağac ,karar ağacının başka methodlarla desteklenip güçlendirilmesi gayet güçlüdür
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


    
# 7. ROC , TPR, FPR değerleri 

y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)
