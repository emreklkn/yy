

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
#aylar bağımsız değişken , satışlar bağımlı değişken
'''
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test =sc.fit_transform(y_test)
#Bu işlemler, veri setinizdeki özellikleri (X_train, X_test) ve hedef değişkenleri (Y_train, Y_test) birbirine benzer bir ölçekte standardize etmeyi veya normalleştirmeyi amaçlar.

'''

#crısp-dm e göre modelling aşamasına geldik
from sklearn.linear_model import LinearRegression #linner kütüphanesinden yani doğrusal olan
lr =LinearRegression()#kütüphanedeki özelliklere erişmek için 
lr.fit(x_train , y_train)             #fit fonksiyonu şuanda  modeli inşaa etmeye çalışıyor 
#buradaki amaç bu verilerle doğrusal bir bağlantı bulunmaya çalışıcak bulunursa geliştirilicek

tahmin = lr.predict(x_test) #predict tahmin ettirme
print(tahmin)

x_train = x_train.sort_index()#bunlar yukarıda random olarak sıralandırmıştık burada grafik için sıralı sıralanmasını sağlıyor

y_train = y_train.sort_index()

#grafik oluşumu trainler eğitilen değerler
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("aylara göre satış")
plt.xlabel("aylar")
plt.ylabel("satış")













