#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:37:29 2018

@author: Brettmccausland
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

def minkowskiDist(x,y,p):
  sum=0
  count=len(x)
  for i in range(count):
    sum = sum + pow((abs(x[i]-y[i])),p)
  fp= 1/p
  return pow(sum,fp)




def knn_classifier(x_test, x_train, y_train, k, p):
    
    return
# main() 
#Question 0: Getting real data  [5%]


#Make sure you can download the dataset, load it into your workspace, 
data=pd.read_csv('BreastCancerWisconsin.csv')
data.drop(data.columns[[0]], axis=1, inplace=True)


#Important : If any data point has missing value(s), document how you 
#are going to handle it, according to what we said in class.
# using the strategy of replacing the missing values with the mean
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
vs=data.iloc[:,[5]].values
imp = imp.fit(vs)
vs=imp.transform(vs)
data.iloc[:,5] = vs
#Question 1: k-Nearest Neighbor Classifier  [50%]
# The second classifier we will see in this assignment
# is the k-Nearest Neighbor (or k-NN) classifier. k-NN is a 
# lazy classification algorithm, which means that it skips the 
# training step and makes a decision during testing time. 
# For the distance calculations, you will use the Lp norm function 
# you implemented in the previous assignment. 
# You should implement the following function:
# y_pred = knn_classifier(x_test, x_train, y_train, k, p)

#Where the inputs are:
#1. x_test = a ‘test data point’ x ‘feature’ matrix containing all 
# the test data points that need to be classified



#2. x_train = a ‘training data point’ x ‘feature’ matrix containing 
# all the training data points that you will compare each test data 
# point against.


#3. y_train = a ‘train data point’ x 1 vector containing the 
# labels for each training data point in x_train 
# (there is obviously a 1-1 correspondence of rows between x_train 
#  and y_train).

#spliting the data

X=data.iloc[:,:-1].values #the feature matrix
Y=data.iloc[:,9].values #the class column

#note Y-test is for later comparision with Y_prediction
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.3)

#4. k = the number of nearest neighbors you select


#5. p  = the parameter of Lp distance as we saw it in Assignment 1.
# The output of the function is:


#1. y_pred = a ‘test data point’ x 1 vector containing the predicted 
# labels for each test data point in x_test (again, there is 
# correspondence between the rows).




















































