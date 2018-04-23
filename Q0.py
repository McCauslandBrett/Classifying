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

def minkowskiDist(x,y,p):
  sum=0
  count=len(x)
  for i in range(count):
    sum = sum + pow((abs(x[i]-y[i])),p)
  fp= 1/p
  return pow(sum,fp)
# main() 
#Question 0: Getting real data  [5%]
#In this assignment you will focus on a dataset that is suitable
#for binary classification: 
#https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
#This link contains a collection of datasets gathered over the years. 
#You are going to work on breast-cancer-wisconsin.data  
# (and its description  breast-cancer-wisconsin.names ).

#Make sure you can download the dataset, load it into your workspace, 
#import the data set

data=pd.read_csv('BreastCancerWisconsin.csv')

data.drop(data.columns[[0]], axis=1, inplace=True)
imp = Imputer(missing_values='nan',strategy='mean',axis=0)
vs=data.iloc[:,5].values
imp = imp.fit(vs)
vs=imp.transform(vs)
#print(data.iloc[:,5])
#imp = Imputer(missing_values='?',strategy='mean',axis=0)
#imp = imp.fit(data[:,5].values)

#data[:,5].values=imp.transform(data[:,5.values])

#and make sure that every data point has entries for all features.
#Important : If any data point has missing value(s), document how you 
#are going to handle it, according to what we said in class.
#using the strategy of replacing the missing values with the mean

#Note : the serial number of each data point is stated as a feature 
#but in this task it is not very useful, so remove it from the feature 
#representation of your data.
#data.drop(data.columns[[0]], axis=1, inplace=True)


#X = dataset.iloc[:,:-1].values      #[take all the lines , except the last one]
#y = dataset.iloc[:,3].values        #Indepedent value vector
#taking care of missing librarys




















































