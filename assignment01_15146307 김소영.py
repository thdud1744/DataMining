# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:18:37 2018

@author: Administrator
"""
# Use the following packages only
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as tdist
from scipy.stats import chi2 
from scipy.stats import stats
import matplotlib.pyplot as plt

def do_linear_regression(X,y):
    reg = LinearRegression()
    reg.fit(X,y)    
    return [reg.intercept_]+list(reg.coef_)

# Question 1
def predict(X,beta):
    ######## CALCULATE ESTIMATED TARGET WITH RESPECT TO X ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUPUT
    # y_pred: 1D list(array) with length n, the estimated target
    # TODO: prediction
    y_pred = []
    #print(beta[1:])
    #print(beta[0])
    y_pred = np.matmul(X,beta[1:])+beta[0]
    return y_pred

# Question 2
def cal_SS(X,y,beta):
    ######## CALCULATE SST, SSR, SSE ########
    # INPUT
    # model: trained linear regression model
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUTPUT
    # SST, SSR, SSE of the trained model
    # TODO: SS
    SST,SSR,SSE=0,0,0
    y_pred=predict(X, beta)    
    SSE=sum((y-y_pred)**2)
    SSR=sum((y_pred-y.mean())**2)
    SST=SSE+SSR
    return (SST,SSR,SSE)

# Question 3
def f_test(X,y,beta,alpha):
    ######## PERFORM F-TEST ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # alpha: significant level
    # OUTPUT
    # f: f-test statistic of the model
    # pvalue: p-value of f-test
    # decision: f-test result 
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    # TODO: F-test
    f=0
    pvalue=0
    decision=None
    n,p=X.shape
    SSR,SSE=cal_SS(X,y,beta)[1:]
    MSR=SSR/p
    MSE=SSE/(n-p-1)
    f=MSR/MSE
    pvalue=1-fdist.cdf(f,p,n-p-1)
    if pvalue<alpha:
        decision=True
    else:
        decision=False
    return (f,pvalue,decision)

# Question 4
def cal_tvalue(X,y,beta):
    ######## CALCULATE T-TEST TEST STATISTICS OF ALL VARIABES ########
    # INPUT
    # model: trained linear regression model
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUTPUT
    # t: array(list) of size (k+1) with the t-test test statisc of variables (the first value is for intercept)
    # TODO: t-test statistics
    t=[]
    SSE=cal_SS(X,y,beta)[-1]
    n,p=X.shape
    MSE=SSE/(n-p-1)
    X=np.c_[np.ones(n), X] 
    XtXinv=np.linalg.inv(np.matmul(X.T,X))
    se=np.sqrt(MSE*np.diag(XtXinv))
    t=beta/se
    return t

# Question 5
def cal_pvalue(t,X):
    ######## CALCULATE P-VALUE OF T-TEST TEST STATISTICS ########
    # INPUT
    # t: array(list) of size (k+1) with the t-test test statisc of variables (the first value is for intercept)
    # X: n by k (n=# of observations, k=# of input variables)
    # OUTPUT
    # pvalue: array(list) of size (k+1) with p-values of t-test (the first value is for intercept)
    # TODO: p-value of t-test
    pvalue=[]
    n,p=X.shape
    pvalue=1-tdist.cdf(np.abs(t),n-p-1)
    return pvalue

# Question 6
def t_test(pvalue,alpha):
    ######## DECISION OF T-TEST ########
    # INPUT
    # pvalue: array(list) of size (k+1) with p-values of t-test (the first value is for intercept)
    # alpha: significance level
    # OUTPUT
    # decision: array(list) of size (k+1) with t-test results of all variables
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    # TODO: t-test 
    decision=[]
    for p in pvalue:
        if p<alpha/2:
            decision.append(True)
        else:
            decision.append(False)
    return decision

# Question 7
def cal_adj_rsquare(X,y,beta):
    ######## CACLULATE ADJUSTED R-SQUARE ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUTPUT
    # adj_rsquare: adjusted r-square of the model
    # TODO: adjusted r-square
    adj_rsquare=0 
    SST,SSR=cal_SS(X,y,beta)[0],cal_SS(X,y,beta)[1]
    rsquare=SSR/SST
    n,p=X.shape
    adj_rsquare=1-(((n-1)/(n-p-1))*(1-rsquare))
    return adj_rsquare

# Question 8
def skew(x):
    ######## CACLULATE Skewness ########
    # INPUT
    # x: 1D list (array)
    # OUTPUT
    # skew: skewness of the array x
    #TODO: calculate skewness
    #ONLY USE numpy
    skew=0
    n=len(x)
    mean=np.mean(x)
    mu3=(1/n)*sum((x-mean)**3)
    sigma3=((1/n)*sum((x-mean)**2))**(3/2)
    skew=mu3/sigma3
    return skew

# Question 9
def kurtosis(x):
    ######## CACLULATE Skewness ########
    # INPUT
    # x: 1D list (array)
    # OUTPUT
    # kurt: kurtosis of the array x
    #TODO: calculate kurtosis
    #ONLY USE numpy
    kurt=0
    n=len(x)
    mean=np.mean(x)
    mu4=(1/n)*sum((x-mean)**4)
    sigma4=((1/n)*sum((x-mean)**2))**(4/2)
    kurt=mu4/sigma4
    return kurt

# Question 10
def jarque_bera(X,y,beta,alpha):
    ######## JARQUE-BERA TEST ########
    # INPUT
    # model: trained linear regression model
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # alpha: significance level
    # OUTPUT
    # JB: Jarque-Bera test statistic
    # pvalue: p-value of the test statistic
    # decision: Jarque-Bera test result 
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    # TODO: Jarque-Bera test
    JB=0
    pvalue=0
    decision=None
    n,p=X.shape
    error=y-predict(X,beta)
    JB=((n-p)/6)*(skew(error)**2+((kurtosis(error)-3)**2/4))
    pvalue=1-chi2.cdf(JB,p-1)
    if pvalue<alpha:
        decision=True
    else:
        decision=False
    return (JB,pvalue,decision)

# Question 11
def breusch_pagan(X,y,beta,alpha):
    ######## BREUSCH-PAGAN TEST ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # alpha: significance level
    # OUTPUT
    # LM: Breusch-pagan Lagrange multiplier statistic
    # pvalue: p-value of the test statistics
    # decision: Breusch-pagan test result 
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    # TODO: Breusch-pagan test
    LM=0
    pvalue=0
    decision=None
    e_2=(y-predict(X,beta))**2
    aux_reg=do_linear_regression(X,e_2)
    n,p=X.shape
    SST,SSR=cal_SS(X,e_2,aux_reg)[0],cal_SS(X,e_2,aux_reg)[1]
    R_sq=SSR/SST
    LM=n*R_sq
    pvalue=1-chi2.cdf(LM,p-1)
    if pvalue<alpha:
        decision=True
    else:
        decision=False
    return (LM,pvalue,decision)

if __name__=='main':
    # LOAD DATA
    data=pd.read_csv('https://drive.google.com/uc?export=download&id=1YPnojmYq_2B_lrAa78r_lRy-dX_ijpCM', sep='\t')
    # INPUT
    X=data[data.columns[:-1]]
    #X=data[['cement','slag','ash','water','superplastic','coarse','fine','age']] 8개.
    # TARGET
    y=data[data.columns[-1]]
    #y=data['strength']
    alpha=0.05
    coefs=do_linear_regression(X,y)
    #################### TEST YOUR CODE ####################
    plt.scatter(X['cement'],y)
    plt.scatter(X['slag'],y)
    plt.scatter(X['ash'],y)
    plt.scatter(X['water'],y)
    plt.scatter(X['superplastic'],y)
    plt.scatter(X['coarse'],y)
    plt.scatter(X['fine'],y)
    plt.scatter(X['age'],y)
    do_linear_regression(X,y)
    predict(X,do_linear_regression(X,y))
    cal_SS(X,y,do_linear_regression(X,y))
    f_test(X,y,do_linear_regression(X,y),alpha)
    cal_tvalue(X,y,do_linear_regression(X,y))
    cal_pvalue(cal_tvalue(X,y,do_linear_regression(X,y)),X)
    t_test(cal_pvalue(cal_tvalue(X,y,do_linear_regression(X,y)),X),alpha)
    cal_adj_rsquare(X,y,do_linear_regression(X,y))
    skew(y-predict(X,do_linear_regression(X,y)))
    kurtosis(y-predict(X,do_linear_regression(X,y)))
    jarque_bera(X,y,do_linear_regression(X,y),alpha)
    breusch_pagan(X,y,do_linear_regression(X,y),alpha)