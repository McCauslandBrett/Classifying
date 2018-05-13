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
 results = knn_classifier(X_train,X_test, Y_train, k, p)

 return

#Question 2: Evaluation  [45%]
  #means

# useage:
# call for all values of k in range 10
# call for all values p in range 3

# postcondition of 1 call:
# Runs Knn 10 times
# finds and the evaluation of each call
# Saves all 10 Accurcy,10 Sensitivity, 10 Specificity readings
# 10 being 1 Accurcy, 1 Sensitivity, 1 Specificity for each data split
def tenFoldEvaluation(data,Accurcy,Sensitivity,Specificity,k,p):


 data = data.sample(frac=1).reset_index(drop=True)
 #split data into 10 folds
 #dataArr is a array of 10 arrays of data tuples
 dataArr = np.array_split(data, 10)

 #stores training tuples
 trainingtuples = []

 for i in range(10):  # i selects the test
   for j in range(10):# j selects the training data
     if(j!=i):        # if not i it becomes part of the traning data
       trainingtuples.append(dataArr[j])

   # Splitting the data into the folds

   # create a table with the training data
   x_train=pd.concat(trainingtuples)  #let x_train = traingtuples collection

   y_train = x_train.iloc[:,9].values #save class labels
   x_train=x_train.iloc[:,:-1].values #get rid of labels

   #Gathering the testing data
   x_test= dataArr[i] # the test data is i
   y_test = x_test.iloc[:,9].values # save the class lables y
   x_test=x_test.iloc[:,:-1].values # get rid of the class labels x

   #ypred[] will return labels array that should match the testing labels

   ypred = knn_classifier(x_train,x_test, y_train, k, p)

   #call and append the evaluation %'s
   Accurcy.append([accuracy(ypred,y_test)])
   Sensitivity.append([sensitivity(ypred,y_test)])
   Specificity.append([specificity(ypred,y_test)])
   trainingtuples.clear()
 return
# postcondition of 1 call:
# Runs Knn 10 times
# finds and the evaluation of each call
# Saves all 10 Accurcy,10 Sensitivity, 10 Specificity readings
# 10 being 1 Accurcy, 1 Sensitivity, 1 Specificity for each data split
# within each fold of the cross-validation, record mean and standard
# deviation of the performance for k = 1:10, for p = 1 and p =2
def Q2_2(data):
#power 1
 Maccurcy_p1 = []
 Mspecificity_p1=[]
 Msensitivity_p1=[]
 #standard deviation
 STDMsensitivity_p1=[]
 STDaccurcy_p1=[]
 STDMspecificity_p1=[]

 # power 2
 Maccurcy_p2 = []
 Mspecificity_p2 =[]
 Msensitivity_p2 =[]
 #standard deviation
 STDMsensitivity_p2 =[]
 STDaccurcy_p2 =[]
 STDMspecificity_p2 =[]

 #shuffle data

 # 30 times
 for k in range(1,11): #for 10 k nieghbors
  for p in range(1,3):  #for 2 powers
    tenFoldEvaluation(data,Accurcy,Sensitivity,Specificity,k,p)
    if(p==1):
    #MEANS
     Maccurcy_p1.append(np.mean(Accurcy))#([accuracy(ypred,y_test),k,p])
     Msensitivity_p1.append(np.mean(Sensitivity))
     Mspecificity_p1.append(np.mean(Specificity))
     #STANDARD DEVIATIONS
     STDaccurcy_p1.append(np.std(Accurcy))
     STDMsensitivity_p1.append(np.std(Sensitivity))
     STDMspecificity_p1.append(np.std(Specificity))
    else:
     #MEANS
     Maccurcy_p2.append(np.mean(Accurcy))#([accuracy(ypred,y_test),k,p])
     Msensitivity_p2.append(np.mean(Sensitivity))
     Mspecificity_p2.append(np.mean(Specificity))
     #STANDARD DEVIATIONS
     STDaccurcy_p2.append(np.std(Accurcy))
     STDMsensitivity_p2.append(np.std(Sensitivity))
     STDMspecificity_p2.append(np.std(Specificity))   
   
    Accurcy.clear()
    Sensitivity.clear()
    Specificity.clear()

 k_list=[1,2,3,4,5,6,7,8,9,10]
 Xlabel='nieghbors'
 Ylabel='performance'
 #---------------------------Accurcy-----------------------------
 
 #       ----------------- Power 1 -----------
 
 plt.errorbar(x= k_list, y=Maccurcy_p1,yerr=STDaccurcy_p1)
 plt.title('Accurcy Power 1')
 plt.xlabel(Xlabel) 
 plt.ylabel(Ylabel)
 plt.savefig('AccurcyPower1')
 plt.close()
 #               --------- Power 2 -----------

 plt.errorbar(x= k_list, y=Maccurcy_p2,yerr=STDaccurcy_p2)
 plt.title('Accurcy Power 2')
 plt.xlabel(Xlabel) 
 plt.ylabel(Ylabel)
 plt.savefig('AccurcyPower2')
 plt.close()

 #---------------------------Sensitivity-----------------------------
 
 #       ----------------- Power 1 -----------
 
 plt.errorbar(x= k_list, y=Msensitivity_p1,yerr=STDMsensitivity_p1)
 plt.title('Sensitivity Power 1')
 plt.xlabel(Xlabel) 
 plt.ylabel(Ylabel)
 plt.savefig('SensitivityPower1')
 plt.close()
 #               --------- Power 2 -----------

 plt.errorbar(x= k_list,y=Msensitivity_p2,yerr=STDMsensitivity_p2)
 plt.title('Sensitivity Power 2')
 plt.xlabel(Xlabel) 
 plt.ylabel(Ylabel)
 plt.savefig('SensitivityPower2')
 plt.close()

 #---------------------------Specificity-----------------------------
 
 #                   -------- Power 1 -------
 
 plt.errorbar(x= k_list, y=Mspecificity_p1,yerr=STDMspecificity_p1)
 plt.title('Specificity Power 1')
 plt.xlabel(Xlabel) 
 plt.ylabel(Ylabel)
 plt.savefig('SpecificityPower1')
 plt.close()
 #                   -------- Power 2 --------

 plt.errorbar(x= k_list,y=Mspecificity_p2,yerr=STDMspecificity_p2)
 plt.title('Specificity Power 2')
 plt.xlabel(Xlabel) 
 plt.ylabel(Ylabel)
 plt.savefig('SpecificityPower2')
 plt.close()

 return

def Q2_1(data):
 Accurcy=[]
 Sensitivity=[]
 Specificity=[]

 #standard deviation

 k=3
 p=2
 tenFoldEvaluation(data,Accurcy,Sensitivity,Specificity,k,p)
 #MEANS
 Maccurcy= np.mean(Accurcy)#([accuracy(ypred,y_test),k,p])
 Msensitivity= np.mean(Sensitivity)
 Mspecificity = np.mean(Specificity)
 #STANDARD DEVIATIONS
 STDaccurcy = np.std(Accurcy)
 STDsensitivity = np.std(Sensitivity)
 STDspecificity = np.std(Specificity)
 print('Maccurcy:',Maccurcy)
 print('Msensitivity:',Msensitivity)
 print('Mspecificity:',Mspecificity)
 
 print('STDaccurcy:',STDaccurcy)
 print('STDsensitivity:',STDsensitivity)
 print('STDspecificity:',STDspecificity)
 
 return
# -------------------MAIN()--------------------------
Accurcy=[]
Sensitivity=[]
Specificity=[]

data = Q0()
Q1(data)
Q2_1(data)
Q2_2(data)










#print(results)
