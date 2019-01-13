# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:52:58 2018

@author: soyeo
"""
from sklearn.cluster import DBSCAN
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

x1,y1=datasets.make_moons(n_samples=200, noise=0.1)
x2,y2 = datasets.make_circles(n_samples=200, noise=0.1, factor=0.5)


plt.scatter(x1[:,0], x1[:,1], c=y1)
plt.scatter(x2[:,0], x2[:,1], c=y2)

cls1 = DBSCAN(eps=0.15, min_samples=3)
cls1.fit(x1)
cls_label=cls1.labels_

set(cls_label)
cls1.core_sample_indices_

##??????
plt.scatter(x1[cls_label!=-1,0], x1[cls_label!=-1,1], c=cls_label[cls_label!=-1])  
plt.scatter(x1[cls_label==-1,0], x1[cls_label==-1,1], marker='x') 

import pandas as pd 

data=pd.read_csv(r'C:\Users\soyeo\Downloads\Chicago_Crimes_2017.csv')

from gmplot import gmplot
from geopy.distance import geodesic
data.head()

X=data[data['Primary Type']=='ROBBERY'] [['Latitude','Longitude']].values
m=gmplot.GoogleMapPlotter(41.832621, -87.658502, 11)
m.scatter(X[:,0],X[:,1], size=40, marker=False)
m.heatmap(X[:,0],X[:,1])
m.draw(r'C:\Users\soyeo\Downloads\robbery.html')



def cal_dist(x,y):
    return geodesic(x,y).km

cls2=DBSCAN(eps=0.15, min_samples=50, metric=cal_dist)

cls2.fit(X)

iris=datasets.load_iris()
X=iris.data
y=iris.target
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, silhouette_score


from sklearn.cluster import KMeans

cls3=KMeans(n_clusters=2) #클러스터 수 다르면 rand score도 달라진다
cls3.fit(X)
l=cls3.labels_
adjusted_rand_score(y, l)
homogeneity_score(y,l)
completeness_score(y,l)


silhouette_score(X,l)
plt.scatter(X[:,0],X[:,1],c=y)#보라는 잘 구분된 반면, 노랑이와 초록이는 뒤섞여있다!!


s=[]
for k in range(2,10):
    cls3.n_clusters=k
    cls3.fit(X)
    s.append(silhouette_score(X,cls3.labels_))

plt.plot(range(2,10),s)



import matplotlib.image as im

sunflower=im.imread(r'C:\Users\soyeo\Downloads\sunflower.jpg')
#왕신기하게도 이미지가 숫자로저장됨,,,

X=np.c_[sunflower[:,:,0].ravel(),sunflower[:,:,1].ravel(),sunflower[:,:,2].ravel()]

X.shape

cls4=KMeans(n_clusters=3)

cls4.fit(X)
cls4.cluster_centers_
l=cls4.labels_

newX=np.zeros(X.shape)

for i in range(3):
    newX[l==i,:]=cls4.cluster_centers_[i]
    
newX=np.reshape(newX,sunflower.shape)
newX=newX.astype('uint8')

plt.imshow(newX) 
plt.axis('off')









