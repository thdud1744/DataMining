# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:20:43 2018

@author: soyeo
"""


from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt

d1X,d1y=datasets.make_blobs(n_samples=200)

plt.scatter(d1X[:,0],d1X[:,1], c=d1y)

cls1=KMeans(n_clusters=2) #3, 4로 해도 okay
cls1.fit(d1X)
l1=cls1.labels_
plt.scatter(d1X[:,0],d1X[:,1], c=l1)

d2X,d2y=datasets.make_moons(n_samples=200, noise=0.2)
plt.scatter(d2X[:,0],d2X[:,1],c=d2y)

cls1=KMeans(n_clusters=2)
cls1.fit(d2X)
l1=cls1.labels_
plt.scatter(d1X[:,0],d1X[:,1], c=l1)

from sklearn.cluster import AgglomerativeClustering

cls2=AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
cls2.fit(d1X)
l2=cls2.labels_
plt.scatter(d1X[:,0],d1X[:,1], c=l2)

cls2.children_
#result of clustering.!!!


from scipy.cluster import hierarchy
from scipy.spatial import distance
import numpy as np

dist=distance.pdist(d1X) 
#d(1,2), d(1,3), d(2,3)
distmat=distance.squareform(dist) #200X200
#대각선은 다 0이다. 왜냐면 자기 자신과 자신의 거리라서

cls2=AgglomerativeClustering(n_clusters=3, affinity="precomputed", linkage="complete")
cls2.fit(distmat)
l2=cls2.labels_
plt.scatter(d1X[:,0],d1X[:,1], c=l2)

cls2.children_

Z=hierarchy.linkage(d1X, method='single', metric='euclidean')
#이Z를 눌러서 칼럼들을 잘 살펴봐라!

l3=hierarchy.cut_tree(Z,n_clusters=[3,4,5])##클러스터 여러개 지정하는것을 한 번에 할 수 있다.!!

hierarchy.dendrogram(Z)

d1X,d1y=datasets.make_blobs(n_samples=50)
plt.scatter(d1X[:,0],d1X[:,1], c=d1y)


Z=hierarchy.linkage(d1X, method='complete', metric='euclidean')
hierarchy.dendrogram(Z)


import pandas as pd

metro=pd.read_csv(r'C:\Users\soyeo\Downloads\METRO_ONOFF.csv', engine='python', index_col=0)
cls4=KMeans(n_clusters=4)
cls4.fit(metro)
np.bincount(cls4.labels_)
len(metro)
center=cls4.cluster_centers_
center.shape


for i in range(4):
    plt.plot(range(24), center[i][:24], label=str(i))

plt.legend()


for i in range(4):
    plt.plot(range(24), center[i][24:], label=str(i))

plt.legend()


X=metro.divide(metro.max(1),axis=0)
 #각 row를 각 row별 Max값으로 나눈다.
 
cls4.fit(X)
center=cls4.cluster_centers_
for i in range(4):
    plt.plot(range(24), center[i][:24], label=str(i))

plt.legend()


for i in range(4):
    plt.plot(range(24), center[i][24:], label=str(i))

plt.legend()

















