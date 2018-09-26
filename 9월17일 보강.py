# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:07:28 2018

@author: soyeo
"""
#############WEEEK 2
#run by line = F9
import pandas as pd
import numpy as np

#판다는 첫 로우가 칼럼네임인 줄 안다. 
#csv 는 컴마로 나누어져 있지만 바꿀 수 있다.
df=pd.read_csv(r'C:\Users\soyeo\OneDrive\문서\3학년2학기\Data Mining\icecream_sale.txt', sep='\t', names=['temp', 'sales'], dtype={'sales':'float'})

df.head() #return top 5
df.tail() #return bottom 5

df.head(10) # return top 10

df.dtypes # check data type

#read로 데이터 생성할 수 있지만 직접 다 칠 수도..
df=pd.Series([1,2,3,4,5])
df

df=pd.DataFrame([[1,2],[3,4]])
df

df=pd.DataFrame([[1,2],[3,4]], columns =['X1','X2'], index=['A', 'B'])
df

df['X1'] # return specific column name 'X1'

df[['X1', 'X2']] #select more than 1 column

df.loc['A','X2'] #access certain value using the name

df.loc['A',['X1','X2']] 

df.iloc[0,0]
df.iloc[-1,-1]

df.index #return name of the index
df.columns #return name of the column
df.values # return only values

df['X1'].mean()
df['X1'].std()
df['X1'].var()
df['X1'].quantile(0.5) #median 

df['X1'].describe()

df.sort_values('X1', ascending=False)

df=pd.read_csv(r'C:\Users\soyeo\Downloads\icecream_sale.txt', sep='\t', names=['temp', 'sales'], dtype={'sales':'float'})


df['temp']>20 #모든 값들의 T/F값을 보여준다,
df[df['temp']>20] #True인 값들만 골라서 표로 보여준다. SQL의 where조건과 같다.

df.iloc[0,1]=100 #원래 215. 이렇게 값 변경할 수 있다.

df=pd.DataFrame([[1,np.nan],[3,4]], columns =['X1','X2'], index=['A', 'B'])
df
df.dropna() #null 값인 것들을 무시한다. 정보를 잃어버릴 수 있다.
df.fillna(10) #모든 null 값을 다른 값으로 대체해준다.

#####WEEK3 LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
#import sklearn 이렇게 임포트하면 밑에서 쓸때마다 LinearRegression 써줘야함. 따라서 specific algorithm을 임포트 해주는 것이 좋다.

df=pd.read_csv(r'C:\Users\soyeo\Downloads\icecream_sale.txt', sep='\t', names=['temp', 'sales'], dtype={'sales':'float'})

reg=LinearRegression()
#fit_intercept조건 : 베타지로가 0일때...????? 
#fit 메소드

X=df['temp']
y=df['sales']

reg.fit(X,y) #요러케 하면 에러남 fit? to train model certain data set. calculating beta.  y = b0 + b1X1+ b2X2 + ...bpXp

X=df[['temp']] 
df['temp'].toframe()
df['temp'].reshape((-1,1))

reg.fit(X,y)#위에 셋 중에 하나를 한 다음에 하면 에러 안남

reg.coef_ #변수 한개니까 이것도 한개일 뿐이다. 5개 있었으면 5개.
reg.intercept_ #= 얘가 바로 beta 0

#따라서 세일즈 = -159 + 0.09*Xtemp
X*reg.coef_+reg.intercept_

#np.array([1,2,3])*np.array([3,4,5]) #이건 그냥 숫자곱이지 행렬 곱이 아니다. 행렬곱은 다른 연산 사용해야한다.

reg.predict(X)

petrol=pd.read_csv(r'C:\Users\soyeo\Downloads\petrol_consumption.txt', sep='\t', names=['tax', 'income', 'highway', 'driver', 'petrol'])

X=petrol[['tax', 'income', 'highway', 'driver']]
y=petrol['petrol']

reg.fit(X,y)

reg.coef_ #여기서는 칼럼이 4개니까 4개 나옴
reg.intercept_ #얘가 베타 0


#넘파이 사용한 베타 햇 구하기 (XtX)-1Xty
np.zeros((3,3)) #모두 0인 3바이3행렬 구한다.
np.ones((2,1)) #모두 1인 2바이 1행렬 구한다.

n,p=X.shape

X=np.c_[np.ones(n), X] #c_는 두개의 칼럼을 붙일 때 사용한다. 따라서 여기는 모두 1인 칼럼벡터를 X에 붙인것.

X.T # transpose

XtX=np.matmul(X.T,X) #matmul이 바로 행렬 곱 계산하는 것이다. 요렇게 하면 5바이 5행렬이 나온다.

XtXinv=np.linalg.inv(XtX) #linear algebra -> inverse 역행렬 구하기. 역시 5바이5

beta=np.matmul(np.matmul(XtXinv, X.T), np.reshape(y.to_frame(),(-1,1))) #이것이 바로 베타들이다.

beta[0]

reg.intercept_

#estimated y = y햇
y_pred=np.matmul(X,beta)

y_pred=y_pred.flatten() #to calculate squared error thing.

y-y_pred #error

SSE=sum((y-y_pred)**2)#sum of sqared error

SSR=sum((y_pred-y.mean())**2)

MSR=SSR/p
MSE=SSE/(n-p-1)

f=MSR/MSE
f


from scipy import stats

stats.f.cdf(f, p, n-p-1)

1-stats.f.cdf(f, p, n-p-1)

np.diag(XtXinv) #0,0 1,1, 2,2, 3,3 등을 보여줌
se=np.sqrt(MSE*np.diag(XtXinv)) #standard deviation

t=beta.flatten()/se

1-stats.t.cdf(abs(t[0]), n-p-1) #얘는 0.05보다 작다. 아웃풋을 믿을만 하다.

1-stats.t.cdf(abs(t[3]), n-p-1) #얘는 0.05보다 크다!!@!!!!!!이 말인즉슨 이건 베타3=0이란 뜻. 왜냐면 아웃풋을 믿을 수 없으니..

#r square = ssr/sst = 1-(sse/sst)

SSR/(SSR+SSE)

X=petrol[['tax', 'income', 'highway', 'driver']]
reg.score(X,y) #SSR/(SSR+SSE)이것과 같은 값을 갖는다.
#score은 항상 r^2을 계산한다. accuracy.




