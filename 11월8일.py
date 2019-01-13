# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 13:44:06 2018

@author: soyeo
"""

from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

d1=pd.read_csv(r'C:\Users\soyeo\Downloads\two_normal.txt', engine='python', sep='\t', names=['X1', 'X2', 'y'])

plt.scatter(d1['X1'],d1['X2'],c=d1['y'])

#clf1=SVC(kernel='linear',C=1) #이거는 62개 나옴
clf1=SVC(kernel='linear',C=10) #이거는 60개 나옴

clf1.fit(d1[['X1','X2']], d1['y'])

clf1.support_

clf1.n_support_

clf1.dual_coef_ 
#alpha!! only non zeros

len(clf1.dual_coef_[0])
#따라서 62개 뿐이다.

clf1.coef_ #w
clf1.intercept_ #b

#우리가 그려야 하는 것은 wT*X + b (sklearn) pdf예제에서는 -b 사용함.
w1=clf1.coef_[0][0]
w2=clf1.coef_[0][1]
b=clf1.intercept_[0]

import numpy as np

xx=np.linspace(-2,5,100)
yy=-w1/w2*xx-b/w2

#두개 같이 그리시오.
plt.scatter(d1['X1'],d1['X2'],c=d1['y'])
plt.plot(xx,yy,'k')

#SVM
#Logistic Regression
#두 개 비교!!


#kernel function
from sklearn import datasets
X,y=datasets.make_moons(n_samples=200, noise=0.17)

plt.scatter(X[:,0],X[:,1],c=y)

clf1=SVC(kernel='linear',C=1)
clf1.fit(X,y)

w1=clf1.coef_[0][0]
w2=clf1.coef_[0][1]
b=clf1.intercept_[0]

xx=np.linspace(-1.5,2,50)
yy=-w1/w2*xx-b/w2
#두개 같이 그리시오.
plt.scatter(X[:,0],X[:,1],c=y)
plt.plot(xx,yy,'k')

clf2=SVC(kernel='rbf', C=1, gamma=1)
clf2.fit(X,y)

#Error occurs coef_ is only available when using a linear kernel
#clf2.coef_
clf2.n_support_
clf2.support_
clf2.dual_coef_

y_pred=clf2.predict(X) #0 또는 1 클래스로 나뉘어짐.
y_df=clf2.decision_function(X) #실제 소수점 값들.

#generate grid to draw 곡선.
xx,yy=np.meshgrid(np.linspace(-1.5,2.5,100), np.linspace(-1,1.5,100))

#ravel() example
a=np.linspace(1,9,9).reshape((3,3))
a.ravel()#3 by 3 became 1 by 9


zz=np.c_[xx.ravel(),yy.ravel()]
#flatten array.
zz_df=clf2.decision_function(zz)

plt.contourf(xx,yy,zz_df.reshape(xx.shape))
plt.scatter(X[:,0],X[:,1],c=y)
#different color means different height.


plt.contourf(xx,yy,zz_df.reshape(xx.shape),levels=[zz_df.min(),0,zz_df.max()])
plt.scatter(X[:,0],X[:,1],c=y)
#3 different region!??


#이렇게 하면 아주 달라진다!(감마가 작아짐!)
clf2=SVC(kernel='rbf', C=1, gamma=0.1)
clf2.fit(X,y)
zz_df=clf2.decision_function(zz)

plt.contourf(xx,yy,zz_df.reshape(xx.shape))
plt.scatter(X[:,0],X[:,1],c=y)
#different color means different height.

plt.contourf(xx,yy,zz_df.reshape(xx.shape),levels=[zz_df.min(),0,zz_df.max()])
plt.scatter(X[:,0],X[:,1],c=y)
#3 different region!??



#이렇게 하면 아주 달라진다!(감마가 커짐!)
clf2=SVC(kernel='rbf', C=1, gamma=10)
clf2.fit(X,y)
zz_df=clf2.decision_function(zz)

plt.contourf(xx,yy,zz_df.reshape(xx.shape))
plt.scatter(X[:,0],X[:,1],c=y)
#different color means different height.

plt.contourf(xx,yy,zz_df.reshape(xx.shape),levels=[zz_df.min(),0,zz_df.max()])
plt.scatter(X[:,0],X[:,1],c=y)
#3 different region!??



##OVER-FITTING : MODEL is TOO COMPLICATED
#이렇게 하면 아주 달라진다!(감마가 매우커짐!)
clf2=SVC(kernel='rbf', C=1, gamma=100)
clf2.fit(X,y)
zz_df=clf2.decision_function(zz)

plt.contourf(xx,yy,zz_df.reshape(xx.shape))
plt.scatter(X[:,0],X[:,1],c=y)
#different color means different height.

plt.contourf(xx,yy,zz_df.reshape(xx.shape),levels=[zz_df.min(),0,zz_df.max()])
plt.scatter(X[:,0],X[:,1],c=y)
#3 different region!??



#kernel : poly degree:5
clf2=SVC(kernel='poly', C=1, degree=5)
clf2.fit(X,y)
zz_df=clf2.decision_function(zz)

plt.contourf(xx,yy,zz_df.reshape(xx.shape))
plt.scatter(X[:,0],X[:,1],c=y)
#different color means different height.

plt.contourf(xx,yy,zz_df.reshape(xx.shape),levels=[zz_df.min(),0,zz_df.max()])
plt.scatter(X[:,0],X[:,1],c=y)
#3 different region!??

#kernel : poly degree:5
clf2=SVC(kernel='poly', C=1, degree=100)
clf2.fit(X,y)
zz_df=clf2.decision_function(zz)

plt.contourf(xx,yy,zz_df.reshape(xx.shape))
plt.scatter(X[:,0],X[:,1],c=y)
#different color means different height.

plt.contourf(xx,yy,zz_df.reshape(xx.shape),levels=[zz_df.min(),0,zz_df.max()])
plt.scatter(X[:,0],X[:,1],c=y)
#3 different region!??
#이건 꽤 다르다... 따라서 SVM을 쓸 때는 적절한 클래시파이어 선택하는 것이 중요하다.



scale=100

#엥 여기는 잘 모르겠당...
xx,yy=np.meshgrid(np.linspace(-1.5*scale,2.5*scale,100), np.linspace(-1*scale,1.5*scale,100))
Xnew=X*scale
zz=np.c_[xx.ravel(),yy.ravel()]
clf2=SVC(kernel='rbf', C=1, gamma=0.001)
clf2.fit(X,y)
zz_df=clf2.decision_function(zz)

plt.contourf(xx,yy,zz_df.reshape(xx.shape))
plt.scatter(X[:,0],X[:,1],c=y)

plt.contourf(xx,yy,zz_df.reshape(xx.shape),levels=[zz_df.min(),0,zz_df.max()])
plt.scatter(X[:,0],X[:,1],c=y)

