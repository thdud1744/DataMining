# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:47:40 2018

@author: Administrator
"""
# Use the following packages only
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def logistic_reg(X,y):
    clf=LogisticRegression()
    clf.fit(X,y)
    return list(clf.intercept_)+list(clf.coef_[0])

# QUESTION 1
def cal_logistic_prob(X,y,beta):
    ######## CALCULATE PROBABILITY OF THE CLASS ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target (Assumption: binary classification problem)
    # beta: array(list) of size (k+1) with the estimated coefficients of variables (the first value is for intercept)
    #        This coefficiens are for P(Y=k) where k is the larger number in output target variable
    # OUTPUT
    # p: probability of P(Y=k) where k is the larger number in output target variable
    
    # TODO: calculate proability of the class with respect to the given X for logistic regression
    p=[]
    p=1/(1+np.exp(-np.matmul(X,beta[1:])-beta[0]))
    return p

# QUESTION 2
def cal_logistic_pred(y_prob,cutoff,classes):
    ######## ESTIMATE OUTPUT CLASS ########
    # INPUT
    # y_prob: probability of P(Y=k) where k is the larger number in output target variable
    # cutoff: threshold for decision
    # classes: labels of classes
    # OUTPUT
    # y_pred: array(list) with the same size of y_prob, estimated output class 
    
    # TODO: estimate output class based on y_prob and cutoff (logistic regression)
    # if probability>cutoff → classes[1] else classes [0]
    y_pred=[]
    for i in range(0,len(y_prob)):
        if (y_prob[i]>cutoff):
            y_pred.append(classes[1])
        else:
            y_pred.append(classes[0])         
    return y_pred

# QUESTION 3    
def cal_acc(y_true,y_pred):
    ######## CALCULATE ACCURACY ########
    # INPUT
    # y_true: array(list), true class
    # y_pred: array(list), estimated class
    # OUPUT
    # acc: accuracy
    
    # TODO: calcuate accuracy
    acc=0
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    for i in range(0,len(y_true)):
        if(y_true[i]==y_pred[i]):
            acc=acc+1
    acc=acc/len(y_true)
    return acc

# QUESTION 4   
def BNB(X,y): 
    ######## BERNOULLI NAIVE BAYES ########
    # INPUT 
    # X: n by p array (n=# of observations, p=# of input variables)
    # y: output (len(y)=n, categorical variable)
    # OUTPUT
    # pmatrix: 2-D array(list) of size c by p with the probability p_ij where c is number of unique classes in y
        
    # TODO: Bernoulli NB
    pmatrix=[] 
    y=np.array(y)
    n,p=X.shape
    mean=X.mean(0)
    new_X=(X>mean)*1
    new_X=np.array(new_X)
    X_1=[]
    X_2=[]
    unique_y=np.unique(y)
    for i in range(0,n):
        if(y[i] == unique_y[0]):
            X_1.append(new_X[i])
        if(y[i] == unique_y[1]):
            X_2.append(new_X[i])     
    X_1 = pd.DataFrame(X_1)
    X_2 = pd.DataFrame(X_2) 
    sum_1=[]
    sum_2=[]  
    for i in range(0,X_1.shape[1]):
        sum_1.append(X_1[i].sum())
        sum_1[i]=sum_1[i]/X_1.shape[0]        
    for i in range(0,X_2.shape[1]):
        sum_2.append(X_2[i].sum())
        sum_2[i]=sum_2[i]/X_2.shape[0] 
    pmatrix.append(sum_1)
    pmatrix.append(sum_2)
    return pmatrix

# QUESTION 5
def cal_BNB_prob(X,prior,pmatrix):
    ######## CALCULATE PROBABILITY OF THE CLASS ########
    # INPUT 
    # X: n by p array (n=# of observations, p=# of input variables)
    # priors: 1D array of size c where c is number of unique classes in y, prior probabilities for classes
    # pmatrix: 2-D array(list) of size c by p with the probability p_ij where c is number of unique classes in y
    # OUTPUT
    # p: n by c array, p_ij stores P(y=cj|X_i)
       
    # TODO: calculate proability of the class with respect to the given X for Bernoulli NB
    p=[]
    pmatrix=np.array(pmatrix)
    mean=X.mean(0)
    new_X=(X>mean)*1
    new_X=np.array(new_X)
    for i in range(0,len(X)):
        new_X[i]=new_X[i].reshape(1,-1)
        a=((pmatrix**new_X[i])*(1-pmatrix)**(1-new_X[i])).prod(axis=1)*prior
        a=a/a.sum()
        p.append(a)
    return p

# QUESTION 6
def cal_BNB_pred(y_prob,classes):
    ######## ESTIMATE OUTPUT CLASS ########
    # INPUT
    # y_prob: probability of P(Y=k) where k is the larger number in output target variable
    # classes: labels of classes
    # OUTPUT
    # y_pred: array(list) with the same size of y_prob, estimated output class 
    
    # TODO: estimate output class based on y_prob (Bernoulli NB)
    y_pred=[]
    for i in range(0,len(y_prob)):
        if(y_prob[i][0] >y_prob[i][1]):
            y_pred.append(classes[0])
        else:
            y_pred.append(classes[1])
    return y_pred
    
# QUESTION 7
def euclidean_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Euclidean distance between a and b
    
    # TODO: Euclidean distance
    d=0
    a=np.array(a)
    b=np.array(b)
    d=np.sqrt(np.sum((a-b)**2))
    return d

# QUESTION 8
def manhattan_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Manhattan distance between a and b
    
    # TODO: Manhattan distance
    d=0
    a=np.array(a)
    b=np.array(b)
    d=np.sum(np.abs(a-b))
    return d

# QUESTION 9
def knn(trainX,trainY,testX,k,dist):
    ######## K-NN Classification ########
    # INPUT 
    # trainX: training input dataset, n by p size 2-D array
    # trainY: training output target, 1-D array with length of n
    # testX: test input dataset, m by p size 2-D array
    # k: the number of the nearest neighbors
    # dist: distance measure function
    # OUTPUT
    # y_pred: predicted output target of testX, 1-D array with length of m
    #         When tie occurs, the final class is select in alpabetical order
    #         EX) if "A" ties "B", select "A" and if "2" ties "4", select 2
    
    # TODO: k-NN classification
    y_pred = []
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    testX=np.array(testX)
    for i in range(0,len(testX)):
        distances={}
        for j in range(0,len(trainX)):
            distances[j]=dist(testX[i],trainX[j])            
        order_dist=np.array(list(distances.items()))
        distances=order_dist[order_dist[:,1].argsort()]    
        k_neigh= [trainY[int(distances[l][0])] for l in range(0,k)]
        if(k_neigh.count(1)>k_neigh.count(2)):
            classi=1 
        else:
            classi=2
        y_pred.append(classi)
    return y_pred

if __name__=='main':
    data=pd.read_csv(r'https://drive.google.com/uc?export=download&id=1QhUgecROvFY62iIaOZ97LsV7Tkji4sY4',names=['ID','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Y'])
    y=data['Y']
    X=data.loc[(y==1)|(y==2),['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']]
    y=y.loc[(y==1)|(y==2)]
    
    trainX,testX,trainY,testY=train_test_split(X,y,test_size=0.2,random_state=11)
    #################### TEST YOUR CODE ####################
    
    #Q2
    cutoff=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    yy=[]
    for i in range(0,len(cutoff)):
        acc=cal_acc(y,cal_logistic_pred(cal_logistic_prob(X,y,logistic_reg(X,y)),cutoff[i],[1,2]))
        yy.append(acc)
    plt.plot(cutoff,yy)
    
    #Q3
    BNB(X,y)

    #Q4    
    logistic_reg(trainX,trainY)
    cal_logistic_prob(testX,testY,logistic_reg(trainX,trainY))
    cal_logistic_pred(cal_logistic_prob(testX,testY,logistic_reg(trainX,trainY)),0.5,[1,2])
    cal_acc(testY,cal_logistic_pred(cal_logistic_prob(testX,testY,logistic_reg(trainX,trainY)),0.5,[1,2]))
    
    BNB(trainX,trainY)
    #trainY's prior---------------------
    unique_trainY=np.unique(trainY)
    trainY=np.array(trainY)
    a=0
    b=0
    for i in range(0,len(trainY)):
        if(trainY[i] == unique_trainY[0]):
            a=a+1
        if(trainY[i] == unique_trainY[1]):
            b=b+1                  
    prior=[a/len(trainY),b/len(trainY)]
    #-----------------------------------
    cal_BNB_prob(testX,prior,BNB(trainX,trainY))
    cal_BNB_pred(cal_BNB_prob(testX,prior,BNB(trainX,trainY)),[1,2])
    cal_acc(testY,cal_BNB_pred(cal_BNB_prob(testX,prior,BNB(trainX,trainY)),[1,2]))
    
    #Q5
    knn(trainX,trainY,testX,3,euclidean_dist)
    cal_acc(testY,knn(trainX,trainY,testX,3,euclidean_dist))
    
    knn(trainX,trainY,testX,5,euclidean_dist)
    cal_acc(testY,knn(trainX,trainY,testX,5,euclidean_dist))
    
    knn(trainX,trainY,testX,7,euclidean_dist)
    cal_acc(testY,knn(trainX,trainY,testX,7,euclidean_dist))
    
    knn(trainX,trainY,testX,3,manhattan_dist)
    cal_acc(testY,knn(trainX,trainY,testX,3,manhattan_dist))
    
    knn(trainX,trainY,testX,5,manhattan_dist)
    cal_acc(testY,knn(trainX,trainY,testX,5,manhattan_dist))
    
    knn(trainX,trainY,testX,7,manhattan_dist)
    cal_acc(testY,knn(trainX,trainY,testX,7,manhattan_dist))
    
    
