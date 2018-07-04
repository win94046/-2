
# coding: utf-8

# In[3]:


from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[4]:


import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# In[5]:


from sklearn.preprocessing import StandardScaler
from keras.models import Sequential


# In[6]:


np.shape(y_test)


# In[7]:


nx_train=np.reshape(x_train,(60000, 28*28))
nx_test=np.reshape(x_test,(10000, 28*28))
nx_train = nx_train/255.0
nx_test = nx_test/255.0
print(nx_train)
#ny_train=np.reshape(y_train,(60000, 28*28))


# In[8]:


pca = PCA(n_components=0.97)
newx_train= pca.fit_transform(nx_train)
newx_test = pca.transform(nx_test)
#newy_train = pca.transform(y_train)
print(newx_train.shape)


# In[9]:


clf = SVC()
clf.fit(newx_train, y_train)


# In[10]:


print("訓練資料辨識率:",np.mean(clf.predict(newx_train) == y_train))


# In[94]:


newx_test


# In[11]:


print("測試資料辨識率:",np.mean(clf.predict(newx_test) == y_test))

