# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:35:57 2018

@author: soyeo
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn import datasets

iris=datasets.load_iris()

X=iris.data
y=iris.target


t1=DecisionTreeClassifier(criterion='gini', min_samples_split=20)
t1.fit(X,y)
y_pred=t1.predict(X)
y_prob=t1.predict_proba(X)


t1.score(X,y)

#DOT (graph description language)

#graph graphname {
#        a -- b -- c;
#        b -- d;
#}

from sklearn.externals.six import StringIO
import pydotplus
from sklearn import tree

dot_file=StringIO()
tree.export_graphviz(t1, out_file=dot_file)
g=pydotplus.graph_from_dot_data(dot_file.getvalue())
# g=graph
g.write_pdf(r'C:\tree.pdf') #permission denied


from IPython.display import Image
Image(g.create_gif()) #안됨 ㅠㅠ

import numpy as np
x=np.random.rand(50,1)*np.pi*2
y=np.sin(x)*5+np.random.normal(size=(50,1))
y=y.flatten()


import matplotlib.pyplot as plt
plt.scatter(x[:,0],y)

t2=DecisionTreeRegressor(max_depth=3)
t2.fit(x,y)

testx=np.linspace(0,np.pi*2,100)
testx=testx.reshape((-1,1))

y_pred=t2.predict(testx)

t3=DecisionTreeRegressor(max_depth=5)
t3.fit(x,y)
y_pred2=t3.predict(testx)


plt.scatter(x[:,0],y)
plt.plot(testx.flatten(),y_pred,'r')
plt.plot(testx.flatten(),y_pred2,'k--')













