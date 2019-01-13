# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:05:51 2018

@author: Administrator
"""
# Use the following packages only
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.misc import comb
from itertools import combinations
import matplotlib.pyplot as plt

# QUESTION 1
def gini(D):
    ######## GINI IMPURITY ########
    # INPUT 
    # D: 1-D array containing different classes of samples
    # OUTPUT    
    # gini: Gini impurity
    
    # TODO: Gini impurity
    proportion = D.nunique() / len(D)
    gini = 1.0 - sum(proportion**2)  
    return gini

# QUESTION 2
def split_gini(x,y):
    ######## FIND THE BEST SPLIT ########
    # INPUT 
    # x: a input variable 
    # y: a output variable
    # OUTPUT
    # split_point: a scalar number for split. Split is performed based on x>split_point or x<=split_point
    # gain: information gain at the split point
    
    # TODO: find the best split
    split_point=0    
    gain =0
    initial_gini=gini(x)
    xval=np.unique(x)
    n=len(y)
    candidate= dict()
    for i in range(len(xval)-1):
        split_point=(xval[i]+xval[i+1])/2
        left=np.where(x<=split_point)[0]
        right=np.where(x>split_point)[0]
        l_gini=gini(pd.DataFrame(left))
        r_gini=gini(pd.DataFrame(right))
        after_gini=(len(left)/n)*l_gini + (len(right)/n)*r_gini
        gain=initial_gini-after_gini
        candidate[split_point] = gain
    found= sorted((value, key) for (key, value) in candidate.items())[-1]
    split_point = found[1]
    gain = found[0]
    return (split_point,gain)

# QEUSTION 3
def kmeans(X,k,max_iter=300):
    ############ K-MEANS CLUSTERING ##########
    # INPUT
    # X: n by p array (n=# of observations, p=# of input variables)
    # k: the number of clusters
    # max_iter: the maximum number of iteration
    # OUTPUT
    # label: cluster label (len(label)=n)
    # centers: cluster centers (k by p)
    ##########################################
    # If average distance between old centers and new centers is less than 0.000001, stop
    
    # TODO: k-means clustering
    label = []
    centers = []
    n, c = X.shape
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    centers = np.random.randn(k, c)*std + mean
    new_centers = np.zeros(centers.shape)
    clusters = np.zeros(n)
    distances = np.zeros((n,k))
    error = np.linalg.norm(new_centers - centers)
    while error != 0:
        for i in range(k):
            distances[:,i] = np.linalg.norm(X - centers[i], axis=1)
        clusters = np.argmin(distances, axis=1)
        centers = new_centers
        for i in range(k):
            new_centers[i] = np.mean(X[clusters ==i], axis=0)
        error = np.linalg.norm(new_centers - centers)    
    label = clusters
    centers = new_centers         
    return (label, centers)

# QUESTION 4
def cal_support(data,rule):
    ######## CALCULATE SUPPORT OF ASSOCIATION RULE ########
    # INPUT
    # data: transaction data, each row contains items
    # rule: array or list with two elements, rule[0] is a set of condition items and rule[1] is a set of result itmes
    # OUTPUT
    # support: support of the rule
    #######################################################
    
    # TODO: support 
    support = 0
    if len(rule) ==1:
        for transaction in data:
            if (rule in transaction):
                support +=1
    else:
        for transaction in data: 
            if set(rule[0]).issubset(set(transaction)):
                support +=1       
    support = support/len(data)
    return support

# QUESTION 5
def cal_conf(data,rule):
    ######## CALCULATE CONFIDENCE OF ASSOCIATION RULE ########
    # INPUT
    # data: transaction data, each row contains items
    # rule: array or list with two elements, rule[0] is a set of condition items and rule[1] is a set of result itmes
    # OUTPUT
    # confidence: confidence of the rule
    #########################################################
    
    # TODO: confidence
    confidence = 0
    bunmo=0
    bunja = cal_support(data, rule)*len(data)
    for transaction in data: 
        if set(rule[0]).issubset(set(transaction)):
            bunmo +=1
    confidence = bunja/bunmo
    return confidence

# QEUSTION 6
def generate_ck(data,k,Lprev=[]):
    ######## GENERATE Ck ########
    # INPUT
    # data: transaction data, each row contains items
    # k: the number of items in sets
    # Lprev: L(k-1) for k>=2
    # OUTPUT
    # Ck: candidates of frequent items sets with k items
    ##############################
    
    # TODO: Ck    
    if k==1:
        Ck=[]
        for transaction in data:
            for item in transaction:
                if item not in Ck:
                    Ck.append(item)
        return Ck
    else:
        Ck=[]
        identical_items=[]
        n_Lprev=[]
        
        if str(Lprev[0]).find('{') == 0:
            for item in Lprev:
                i=item.split('\'')[1::2]
                n_Lprev.append(i)
        else:
            n_Lprev=Lprev
        
        for item_set in n_Lprev:
            for item in item_set:
                if item not in identical_items:
                    identical_items.append(item)
        candidate_tuple = list(combinations(identical_items, k))
        for item in candidate_tuple:
            Ck.append(set(item))
        return Ck

# QEUSTION 7
def generate_lk(data,Ck,min_sup):
    ######## GENERATE Lk ########
    # INPUT
    # data: transaction data, each row contains items
    # Ck: candidates of frequent items sets with k items
    # min_sup: minimum support
    # OUTPUT
    # Lk: frequent items sets with k items
    ##############################
    
    # TODO: Lk
    # Use cal_support
    Lk=[]
    sup_Ck = dict()
    for item in Ck:
            sup_Ck[str(item)] = cal_support(data, item)
            
    for key in sup_Ck.keys():
        if sup_Ck[key] >= min_sup:
            Lk.append(key)
    return Lk

# QEUSTION 8
def PCA(X,k):
    ######## PCA ########
    # INPUT
    # X: n by p array (n=# of observations, p=# of input variables)
    # k: the number of components
    # OUTPUT
    # components: p by k array, each column corresponds to PC in order. (the first PC is the first column)
    
    # TODO: PCA
    # Hint: use numpy.linalg.eigh
    components = np.array
    cov_mat = np.cov(X.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    components = X.dot(w)
    return components

if __name__=='main':    
    cancer=pd.read_csv('https://drive.google.com/uc?export=download&id=1-83EtpdXI_bNWlWD7v-t_7XLJgnwocxg')
    iris=load_iris()
    trans=pd.read_csv('https://drive.google.com/uc?export=download&id=1F_6wOpWqO-yXfbpfSCXfX6_uV4YhOPqD', index_col=0)
    trans=[x.split(',') for x in trans['Items'].values]
    #################### TEST YOUR CODE ####################
    
    ## 2
    X=cancer.drop(["Diagnosis", "ID"], axis=1)
    y = cancer[["Diagnosis"]]
    b = dict()    
    result_split_gini = []
    for column in list(X):
        x=cancer[[column]]
        sg = split_gini(x,y)
        result_split_gini.append(sg)
        b[column] = sg[1]
    first_split_variable = sorted((value, key) for (key, value) in b.items())[-1]
    
    ## 3
    X=iris.data
    y=kmeans(X, 3, max_iter=300)[0]
    n, c = X.shape
    colors=['orange', 'blue', 'green']
    for i in range(n):
        plt.scatter(X[i, 0], X[i,1], s=7, color = colors[int(y[i])])
   
    ## 4
    cal_support(trans,['a', 'b'])
    cal_conf(trans,['a', 'b'])
    
    cal_support(trans,[['b','c','e'], 'f'])
    cal_conf(trans,[['b','c','e'], 'f'])
    
    cal_support(trans,[['a','c'], ['b','f']])
    cal_conf(trans,[['a','c'], ['b','f']])
    
    cal_support(trans,[['b','d'], 'g'])
    cal_conf(trans,[['b','d'], 'g'])
    
    cal_support(trans,[['b','e'], ['c','f']])
    cal_conf(trans,[['b','e'], ['c','f']])
    
    ## 5
    X = PCA(X, 1)
    y=(iris.target).reshape(150,1)
    result = np.append(X,y,axis=1)
    colors=['orange', 'blue', 'green']
    for i in range(150):
        plt.scatter(result[i, 0], result[i,1], s=7, color = colors[int(result[i,2])])
   
    # Apriori algorithm
    min_sup=0.4    
    Ck=generate_ck(trans,1)
    r=dict()
    for k in range(1,len(Ck)):
        Lk=generate_lk(trans,Ck,min_sup)
        r[k]=[Ck,Lk]
        Ck=generate_ck(trans,k+1,Lk)
        if len(Ck)==0:
            break
    r
