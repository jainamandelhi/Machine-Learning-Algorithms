# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 17:45:35 2019

@author: Aman Jain
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as split

dataset=pd.read_table('C:/Users/Aman Jain/Downloads/fruit_data_with_colors.txt')

label={1:"apple",2:"mandarin",3:"orange",4:"lemon"}

X=dataset[["mass","color_score"]].values
y=dataset.iloc[:,0:1].values

X_train,X_test,y_train,y_test=split(X,y,train_size=0.99)

def KNN(q1,q2):
    mini=10000000
    ans=1
    for i in range(len(X_train)):
        diff=np.sqrt((q1-X_train[i][0])**2+(q2-X_train[i][1])**2)
        if(diff<mini):
            mini=diff
            ans=y[i]
    return ans

ans=KNN(X_test[0][0],X_test[0][1])
label[ans[0]]

'''
Out[1]: 'lemon'
'''