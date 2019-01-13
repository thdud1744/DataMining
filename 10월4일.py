# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:39:51 2018

@author: soyeo
"""

#한번에 실행해야 에러 안난다..
def HappyBirthday(person):
    print('Happy birthday, %s' %(person))
HappyBirthday('Tom')

def sum_nums(a,b):
    return a+b
c=sum_nums(1,2)
c

def math(a,b,c=1):#c인풋 없을떄 자동으로 1이 된다,
    return c*(a+b)
math(1,2,3)
math(9,1)

import numpy as np
def bernoulli(D,p):#결과 이상함...
    #p^y*(1-p)^(1-y)
    L=1
    for d in D:
        L*=(p**(d))*(1-p)**(1-d)
        return L
    
def bernoulli(D,p):#결과 제대로 나옴
    D=np.array(D)
    return np.prod((p**D)*((p-1)**(1-D)))

D=[1,0,1,1,1,0] #1과 0은 각각 다른 결과들을 나타낸다.    

1-np.array(D)

bernoulli(D,0.3)#0.003968...
bernoulli(D,0.6)#0.020735...

np.prod([1,2,3])

np.pi
np.exp(-1)
np.exp(1)

np.log(np.exp(1))
np.log2(2**4)

np.sqrt(3)

import pandas as pd

df=pd.read_csv(r'C:\Users\soyeo\Downloads\height.txt', sep='\t', names=['X','Y'])

df.head()

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

clf=LogisticRegression()

df=df.sort_values(['Y'])
t=(df['Y']=='male')*1

n=len(df)
plt.scatter(range(n), df['X'], c=t)


#P(Y=1 | x) = 1/(1+e^-Xtb)
clf.fit(df[['X']], df['Y'])

clf.coef_

clf.intercept_

#P(Y=0|x)
clf.predict(df[['X']])
#모두 남자로 나온다. 이유는? 확률이 높은것 선택했더니 그렇게 된것.

clf=LogisticRegression(C=10e5)#100,000
clf.fit(df[['X']],df['Y'])

clf.coef_
clf.intercept_
#decrease impact of beta..?

clf.predict(df[['X']])
# 이제서야 여자 남자 섞어서 나온다!! 
# accuracy = num of crrectly classified sample / num of total example

y_pred=clf.predict(df[['X']])

y_pred[:10]#male 2개 나와야되는데 난 전부 female 이다..
df['Y'][:10]# 10개 전부 피메일. 
#accuracy = 8/10 = 0.8

sum(y_pred==df['Y'])/n

clf.score(df[['X']],df['Y'])
#accuracy
y_prob=clf.predict_proba(df[['X']])
y_prob
#female=0 male=1. probability of female in row 1 = 0.96
y_pred[0]# 실제로 female
y_pred[-1]

y_prob[:,0]>0.6

from sklearn.datasets import load_iris
iris=load_iris()
X=iris['data']
y=iris['target']

clf=LogisticRegression()
clf.fit(X,y)
clf.coef_
clf.intercept_

clf=LogisticRegression(multi_class='multinomial') #에러가 나는것이 정상!
clf=LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(X,y)

y_prob=clf.predict_proba(X)
#세 종류중 무슨 종류인지 확률 알려주는 중.





