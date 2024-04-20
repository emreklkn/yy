# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

#verileri 2 parçaya böleliM
x= veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]#2.satırı aldık sadece
X=x.values
Y=y.values

#linner regresyon
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)#ilk öğretme işlemini yapıtyoruz aşağıda da öğrendiğinden yola çıkarak predict diyip tahmin ettireceğiz

plt.scatter( x, y)#görselleştirme
plt.plot(x,lin_reg.predict(x))#öğrenme
plt.show()

#şimdi gelelim polinom regresyona
from sklearn.preprocessing import PolynomialFeatures# bu kütüphane istediğimize polinomal derece verebiliyoruz
poly_reg= PolynomialFeatures(degree=2)# 2.dereceden bir obe oluştur
#burada amaç ilk polinomal olarak tasarlıyıcaz sonra ise doğrusal dünyaya koyucağız
x_poly=poly_reg.fit_transform(x)
print(x_poly)
lin_reg2= LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y)
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(X))
plt.show()


