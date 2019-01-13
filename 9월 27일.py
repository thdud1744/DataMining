# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:32:14 2018

@author: soyeo
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

datapath=r'C:\Users\soyeo\Downloads'

df=pd.read_csv(os.path.join(datapath, 'petrol_consumption.txt'), sep='\t', names=['tax', 'income', 'highway', 'driver', 'petrol'])

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

X=df[['tax', 'income', 'highway', 'driver']]
y=df['petrol']

reg.fit(X,y)





error=y-reg.predict(X)

plt.hist(error, bins=10) #최소에서 최대의 구간이 10개로 쪼개진다.
plt.hist(error, bins=20) #너무 작은 bin수는 부적절. 너무 큰 수는 noise 발생.적절한 수의 bin을 구하는 게 좋다.

stats.probplot(error, dist='norm', plot=plt)
#Q-Q plot!!!

#skewness : Jarque-Bera
stats.skew(error)
stats.kurtosis(error, fisher=True)#default. (normal distribution. c-3한 값을 보여줌)
stats.kurtosis(error, fisher=False)#피피티 기본 공식으로 계산! 

stats.chi2.pdf
stats.chi2.cdf()

e2=error**2
reg.fit(X,e2)
reg.score(X,e2)





reg.coef_

#베타 0
reg.intercept_

#predict 메소드로 예측하기
y_pred=reg.predict(X)



x=np.arange(300, 1001, 100) #300  400  500  600  700  800  900 1000
np.linspace(300, 1000, 8)#array([ 300.,  400.,  500.,  600.,  700.,  800.,  900., 1000.])
print(x)
l=x

plt.scatter(y, y_pred, s=50, c='mediumslateblue')
#모든 플롯은 다 두 종류의 인풋이 필요하당  오 왕신기. 그래프 그려줌
#s = size of dot   c= color
plt.plot(x,l)
#40줄부터 43줄까지 한번에 실행시키면 두 그래프가 겹쳐 보인다.

plt.scatter(df['income'],df['petrol'])
plt.xlabel('income')
plt.ylabel('petrol consumption')
plt.title('scatter plot')
#여기도 46~49한번에 실행

y_pred.mean()
y_pred.std()
y_pred.var()

SSE=sum((y-y_pred)**2)

SSR=sum((y_pred-y_pred.mean())**2)

X.shape
n, p = X.shape

MSE=SSE/(n-p-1)
MSR=SSR/p

f=MSR/MSE
f
#f(df1, df2)
#df1 = MSR -> p
#df2 = MSE -> n-p-1


stats.f.pdf(f, p, n-p-1) 
stats.f.cdf(f, p, n-p-1)
1-stats.f.cdf(f, p, n-p-1) #cumulate density function
#f = 22.706   pdf =2.459  cdf=0.99999 1-cdf = 3.907

Xmat=np.c_[np.ones(n),X]
Xmat.T

XtX=np.matmul(Xmat.T, Xmat)

Xinv=np.linalg.inv(XtX)

reg.coef_[0]

np.diag(Xinv)#(1,1), 2,2 3,3 등 대각선 자리의 수들을 반환한다

t=reg.coef_[0]/np.sqrt(np.diag(Xinv)*MSE)[1]
#t-test
t

(1-stats.t.cdf(np.abs(t),n-p-1))*2
#흠... 어렵누...


reg.fit(df[['income','highway', 'driver']], df['tax'])

1/(1-reg.score(df[['income','highway', 'driver']], df['tax']))


blood=pd.read_csv(os.path.join(datapath, 'bloodpress.txt'), sep='\t', index_col=0)

cov=blood.cov()
cor=blood.corr()

X1=blood[['Age','Weight','BSA','Dur','Pulse','Stress']]
X2=blood[['Age','BSA','Dur','Pulse','Stress']]

y=blood['BP']

reg1=LinearRegression()
reg2=LinearRegression()

reg1.fit(X1,y)
reg2.fit(X2,y)

reg1.coef_#몸무게와 몸표면적(BSA)는 비슷한 경향을 보인다.
reg2.coef_#여기서 봐야할 것은 weight의 유무에 따라 BSA의 coef가 변하는 양상이다.

reg.fit(blood[['Age','BSA','Dur','Pulse','Stress']],blood['Weight'])
1/(1-reg.score(blood[['Age','BSA','Dur','Pulse','Stress']],blood['Weight']))
#얘는 샘플이 너무 작아서 맘에드는 값이 안나온다. 페트롤를 봐랏! 23번째 줄.


