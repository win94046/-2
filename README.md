作業2 是這次的功課

程式說明:
1.
  from keras.datasets import mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
因為sklearn無法加載mnist，所以改用 keras來加載mnist

2. 
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
引用

3.
np.shape(x_train)
np.shape(x_test)
np.shape(y_train)
np.shape(y_test)
先確定資料的型態， 以便了解如何調整成 PCA需要的格式會分別得到
(60000, 28, 28)
(10000, 28, 28)
(60000,)
(10000,)

4.
nx_train=np.reshape(x_train,(60000, 28*28))
nx_test=np.reshape(x_test,(10000, 28*28))
把train 跟test 圖片的像素[28*28]  變成784*1的矩陣
nx_train變成(60000,784) , nx_test變成(10000,784)
5.
nx_train = nx_train/255.0
nx_test = nx_test/255.0
print(nx_train)
把矩陣裡的數都除255 讓圖片變成1~0之間的數
然後再確認一次
 6.
pca = PCA(n_components=0.97)
newx_train= pca.fit_transform(nx_train)
newx_test = pca.transform(nx_test)
print(newx_train.shape)
使用sklearn上的PCA套件，其中參數n_components 設置成0.97 使電腦自己找出特徵個數 n ，使其滿足要求的百分方差比
其中newx_test 可以藉由newx_train找特徵的分法直接抓，然後在看一下狀況
7.
clf = SVC()
clf.fit(newx_train, y_train)
使用sklearn上的svc來訓練
8. 
print("test資料辨識率:",np.mean(clf.predict(newx_test) == y_test))
clf.predict(newx_test)用訓練好的分類器去預測newx_test 
用np.mean 的function去找出辨識率

##############################################################################################################################
結果:

在n_components 設置成0.97的狀況下使用svc 得到的辨識率有0.9656

心得:

這次在寫功課時一開始就遇到了瓶頸，我發現了sklearn不管加載幾次mnist都會出現連線出錯等問題，在最後學長的建議下改用了keras來加載檔案，
之後依然使用sklearn來運行，下載後已經特別幫我們分成了訓練用的資料與測試用的資料直接帶進PCA就好，然後馬上又失敗，用了好久才發現原本的
資料是3維的矩陣，然後使用了np.reshape來把x_train等其他 改變成 張數 X 784的矩陣，再帶進PCA與SVC執行，在這次的作業中，我最印象深刻的
是我的nx_test = nx_test/255.0 ，我前面還因為test 打成train 導致我的辨識率一直是0，最後才找到原來是因為錯字，也讓我學到了一個教訓。
這次的作業讓我學到了一些基本的sklearn 跟keras的使用方法，雖然被不能加載mnist擺了一道哈哈。
 
 
 
 
  
 
  
