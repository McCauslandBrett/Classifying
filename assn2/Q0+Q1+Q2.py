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
import operator
from operator import itemgetter, attrgetter
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

def minkowskiDist(R,Rn,p):
  sum=0
  count=len(R)
  for i in range(count):
    sum = sum + pow((abs(R[i]-Rn[i])),p)
  fp= 1/p
  return pow(sum,fp)
#performance measures: 
def accuracy(ypred,ytest):
 c = 0
 for i in range(len(ytest)):
     if(ytest[i] == ypred[i]):
         c += 1 
 return(c/float(len(ytest)) * 100.0)


def sensitivity(ypred,ytest):
 Tp =0
 Fn=0
 
 for i in range(len(ytest)):
     if( ytest[i]== 4 and ypred[i]== 4):
         Tp += 1 
     elif (ytest[i]== 4 and ypred[i]==2):
         Fn+=1
 return float((Tp /(Tp + Fn) * 100.0))

def specificity(ypred,ytest):
 Tn =0
 Fp=0
 for i in range(len(ytest)):
     if(ytest[i]== 2 and ypred[i]== 2):
         Tn += 1 
     elif(ytest[i]== 2 and ypred[i]==4):
         Fp+=1
 return float((Tn /(Tn + Fp) * 100.0))

def Rowneighbors(X_train, X_test,k,p):
   d = []
   neighbors = []
   trainsize = len(X_train)
    
   for i in range(trainsize):
     dist = minkowskiDist(X_train[i],X_test,p)
     d.append((dist,i))
   #sort by dist 
   d.sort(key=operator.itemgetter(0))
   for i in range(k):
     neighbors.append(d[i])
   return neighbors

def knn_classifier(x_train,x_test, y_train, k, p):
    testsize=len(x_test)
    results=[]
    c1=0
    c2=0
    ypredictions=[]
    #get the nieghbors
    for i in range(testsize):
        r = Rowneighbors(x_train, x_test[i],k,p)
        results.append(r)
    #check what class the nieghbors are
    for i in range(testsize):
        #l is the array of pairs
        l=results[i]
        for d,I in l:
            if(y_train[I]==2):
             c1=c1+1
            else:
             c2=c2+1
        #decide the prediction
        if(c1>c2):    
         ypredictions.append(2)
        else:
         ypredictions.append(4)
        
        c1=0
        c2=0
   
    return ypredictions

#Question 0: Getting real data  [5%]
def Q0():
 
 #import data
 data=pd.read_csv('BreastCancerWisconsin.csv')

 #remove useless column
 data.drop(data.columns[[0]], axis=1, inplace=True)

 #using the strategy of replacing the missing values with the mean
 imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
 vs=data.iloc[:,[5]].values
 imp = imp.fit(vs)
 vs=imp.transform(vs)
 data.iloc[:,5] = vs
    
 return data

def Q1(data):
 X=data.iloc[:,:-1].values #the feature matrix
 Y=data.iloc[:,9].values #the class column
 #note Y-test is for later comparision with Y_prediction
 X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.2)
 # get neighbors for one row
 k=7
 p=2
 #result = Rowneighbors(X_train, X_test[0],k,p)
 results = knn_classifier(X_train,X_test, Y_train, k, p)
    
 return 

#Question 2: Evaluation  [45%]
def Q2(data,Accurcy,Sensitivity,Specificity):
 Maccurcy = []
 Mspecificity=[]
 Msensitivity=[]
 
 STDMsensitivity=[]
 STDaccurcy=[]
 STDMspecificity=[]
 
 for k in range(1,10):
  for p in range(1,3):
   tenFoldEvaluation(data,Accurcy,Sensitivity,Specificity,k,p)
   #MEANS
   Maccurcy.append(np.mean(Accurcy))
   Msensitivity.append(np.mean(Sensitivity))
   Mspecificity.append(np.mean(Specificity))
   #STANDARD DEVIATIONS
   STDaccurcy.append(np.std(Accurcy))
   STDMsensitivity.append(np.std(Sensitivity))
   STDMspecificity.append(np.std(Specificity))
 print(Maccurcy)
 print('d')
 return 

def tenFoldEvaluation(data,Accurcy,Sensitivity,Specificity,k,p):
 #shuffle data
 data = data.sample(frac=1).reset_index(drop=True)
 #split data into 10 folds
 dataArr = np.array_split(data, 10)
 frames = []
 for i in range(10):# i selects the test 
   for j in range(10):# j selects the training data 
     if(j!=i):
       frames.append(dataArr[j])
   x_train=pd.concat(frames)
   y_train = x_train.iloc[:,9].values
   x_train=x_train.iloc[:,:-1].values
   
   x_test= dataArr[i]
   y_test = x_test.iloc[:,9].values
   x_test=x_test.iloc[:,:-1].values
   print(i)#checking outside the innnerloop 
   ypred = knn_classifier(x_train,x_test, y_train, k, p)
   Accurcy.append([accuracy(ypred,y_test),k,p])
   Sensitivity.append([sensitivity(ypred,y_test),k,p])
   Specificity.append([specificity(ypred,y_test),k,p])
   frames.clear()
 return




# -------------------MAIN()--------------------------
 Accurcy=[]
 Sensitivity=[]
 Specificity=[]
 
 
 data = Q0()
 Q1(data)
 Q2(data,Accurcy,Sensitivity,Specificity)
 
 # ------ Accurcy Decomposition -------
 Accurcy.sort(key=operator.itemgetter(2))
 AccurcyTable=pd.DataFrame(list(Accurcy))
 
             # ------ p=1 -------
 Acc_p1=AccurcyTable.iloc[0:90,0].values
 Acc_k_p1=AccurcyTable.iloc[0:90,1].values
 plt.scatter(Acc_k_p1,Acc_p1)
 plt.title('Accurcy power 1')
             # ------ p=2 -------
 Acc_p2=AccurcyTable.iloc[90:180,0].values
 Acc_k_p2=AccurcyTable.iloc[90:180,1].values
 plt.scatter(Acc_k_p2,Acc_p2)
 plt.title('Accurcy power 2')
 
 # ------ Sensitivity Decomposition -------
 
 Sensitivity.sort(key=operator.itemgetter(2))
 SensitivityTable=pd.DataFrame(list(Sensitivity))

             # ------ p=1 -------
 Sens_p1=SensitivityTable.iloc[0:90,0].values
 Sens_k_p1=SensitivityTable.iloc[0:90,1].values
 plt.scatter(Sens_k_p1,Sens_p1, c="g", alpha=0.4,marker='^')
 plt.ylabel('Sensitivity')
 plt.xlabel('k')
 plt.title('Sensitivity')
 plt.plot(Sens_k_p1,Sens_p1, linewidth=2.0)
              # ------ p=2 -------
 Sens_p2=SensitivityTable.iloc[90:180,0].values
 Sens_k_p2=SensitivityTable.iloc[90:180,1].values
 plt.scatter(Sens_k_p2,Sens_p2, c="b", alpha=0.4,marker='o')
 plt.legend()
 #plt.title('Sensitivity power 2')
 
 # ------ Specificity Decomposition ------- 
 Specificity.sort(key=operator.itemgetter(2))
 SpecificityTable=pd.DataFrame(list(Specificity))
              # ------ p=1 -------
 Spec_p1=SpecificityTable.iloc[0:90,0].values
 Spec_k_p1=SpecificityTable.iloc[0:90,1].values
 plt.scatter(Spec_p1,Spec_k_p1)
 plt.title('Specificity power 1')
              # ------ p=2 -------
 Spec_p2=SpecificityTable.iloc[90:180,0].values
 Spec_k_p2=SpecificityTable.iloc[90:180,1].values
 plt.scatter(Spec_p2,Spec_k_p2)
 plt.title('Specificity power 2')













#print(results)







































