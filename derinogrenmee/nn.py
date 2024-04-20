# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""
#UYGULAMA MÜŞTERİLERİN BIRAKIP BIRAKMAYACAĞINNI TESPİT ETMEYE ÇALIŞYORUZ
#1.kutuphaneler 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#veri on isleme

X= veriler.iloc[:,3:13].values#ilk 3 satırı almadık cünkü ezberlemeye gidebilir
Y = veriler.iloc[:,13].values
#cografi bilgileri cinsiyet bilgilerini encoding yapmamız gerekiyor 
from sklearn import preprocessing
le =preprocessing.LabelEncoder()    
X[:,1]=le.fit_transform(X[:,1])#dönüştürüyoruz 1 0 2 vesaile ülkeleri 

#ikinci kolonu encoder edelim cinsiyet 
le2 = preprocessing.LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

#yüksek sayılı kolonları normalize etmek gerekiyor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer # birden fazla column ayrı ayrı veya aynı anda dönüşmesi için 

ohe =ColumnTransformer([("ohe" ,OneHotEncoder(dtype=float),[1])],
                       remainder="passthrough")
X=ohe.fit_transform(X)#ilk eğitiliyor sonra transform ediliyor 
X=X[:,1:]




from sklearn.model_selection import train_test_split

x_train, x_test ,y_train ,y_test =train_test_split(X,Y,test_size=0.33,random_state=0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#veriyi işledik  şimdi yapay sinir ağı kısmı
# burada keraskütüphensi kullanılıcak
import keras 
# bir model belirlenmesi gerekiyor
# sequential bir yapay sinir ağının varlığını belirtlemek içinn  yani kerasa bir yapay sinir ağ  kullanacağımızı belirtiyoruz
from  keras.models import Sequential
from keras.layers import Dense ,Dropout# yapay sinir ağını oluşturabileceğimiz nesneyi içeriyor 

# ilk yapacağımız şey seguentialdan bir obe üretmek yani sınıflandırma algoritmamız olucak
classifier =Sequential()# bilgisayarımızın raminde bir yapay sinir ağımız var 
classifier.add(Dense(32, kernel_initializer='uniform', activation='relu', input_dim=11))

#6 adet nöron , iniform edilirek yani sıfıra yakınlaştırılarak hesaplanıcak
#kullanmamız gereken aktivasyon fonksiyonu belirlencek aktivastyon fonksiyonu ağırlıklar gelince hesaplayan fonksiyon
# aktivasyon fonksiyonumuz relu çünkü 0 ın altında ise 0 , üstündeyse 1 olacak şekilde linner olmasını sağlayacak
# 6 olmasnın sebebi genelde kullanılan yöntem 11 adet sütün var yarısını almak mantıklı 6 adet nöron oluşturmak



# şimdi sıra gizli katman ekleme 
classifier.add(Dense(64, kernel_initializer='uniform', activation='relu'))#ilkinde eklendiğinden input atmaya gerek yok
classifier.add(Dense(64, kernel_initializer='uniform', activation='relu'))#ilkinde eklendiğinden input atmaya gerek yok
classifier.add(Dropout(0.2))
classifier.add(Dense(64, kernel_initializer='uniform', activation='relu'))#ilkinde eklendiğinden input atmaya gerek yok

# çıkış kısmında sigmoid kullanıcaz , loistik olduğu için
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])# derleme ve adam yöntemi ile optimize etmek

classifier.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test))#bağımsız değişkenlerden(xtrain) bağımlı değişkenleri(ytrain) öğren 
y_pred=classifier.predict(X_test)
#loss da optimize ediyor 

y_pred = (y_pred > 0.5)

# confision matrix oluşturarak 0 olanların kaçı doğru kaçı yanlış 1 olanların kaçı doğru kaçı yanlış tespit ediyoruz 1 bırakabilir demek bu problemde
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)
# çıktıya bakılınca bırakmıyıcak kişilerden 2617 den 101 i yanlış gibi












