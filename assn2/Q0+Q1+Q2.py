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
   Accurcy.append([accuracy(ypred,y_test),k,p])
   Sensitivity.append([sensitivity(ypred,y_test),k,p])
   Specificity.append([specificity(ypred,y_test),k,p])
   trainingtuples.clear()
 return
# postcondition of 1 call:
# Runs Knn 10 times
# finds and the evaluation of each call
# Saves all 10 Accurcy,10 Sensitivity, 10 Specificity readings
# 10 being 1 Accurcy, 1 Sensitivity, 1 Specificity for each data split
def tenFoldEvaluationq2(data,Accurcy,Sensitivity,Specificity,k,p):

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

#precondition: Q1
#Accurcy: percentage for
#postcondition: plots = each power,for the evaluations gathered for all 10 folds
def Q2Plots(Accurcy,Sensitivity,Specificity):
 # ------ Accurcy Decomposition -------

 Accurcy.sort(key=operator.itemgetter(2))
 AccurcyTable=pd.DataFrame(list(Accurcy))

             # ------ p=1 -------
 Acc_p1=AccurcyTable.iloc[0:90,0].values
 Acc_k_p1=AccurcyTable.iloc[0:90,1].values
 plt.scatter(Acc_k_p1,Acc_p1)
 plt.title('Accurcy power 1')
 plt.figure()
             # ------ p=2 -------
 Acc_p2=AccurcyTable.iloc[90:180,0].values
 Acc_k_p2=AccurcyTable.iloc[90:180,1].values
 plt.scatter(Acc_k_p2,Acc_p2)
 plt.title('Accurcy power 2')
 plt.figure()

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
 plt.figure()
              # ------ p=2 -------
 Sens_p2=SensitivityTable.iloc[90:180,0].values
 Sens_k_p2=SensitivityTable.iloc[90:180,1].values
 plt.scatter(Sens_k_p2,Sens_p2, c="b", alpha=0.4,marker='o')
 plt.legend()
 plt.figure()
 #plt.title('Sensitivity power 2')

 # ------ Specificity Decomposition -------
 Specificity.sort(key=operator.itemgetter(2))
 SpecificityTable=pd.DataFrame(list(Specificity))
              # ------ p=1 -------
 Spec_p1=SpecificityTable.iloc[0:90,0].values
 Spec_k_p1=SpecificityTable.iloc[0:90,1].values
 plt.scatter(Spec_p1,Spec_k_p1)
 plt.title('Specificity power 1')
 plt.figure();
              # ------ p=2 -------
 Spec_p2=SpecificityTable.iloc[90:180,0].values
 Spec_k_p2=SpecificityTable.iloc[90:180,1].values
 plt.scatter(Spec_p2,Spec_k_p2)
 plt.title('Specificity power 2')
 plt.figure()
 return


# -------------------MAIN()--------------------------
Accurcy=[]
Sensitivity=[]
Specificity=[]

data = Q0()
Q1(data)
#------------------Question 2-------------------------
# within each fold of the cross-validation, record mean and standard
# deviation of the performance for k = 1:10, for p = 1 and p =2
Maccurcy_tuple = []
Mspecificity_tuple=[]
Msensitivity_tuple=[]

#standard deviation
STDMsensitivity_tuple=[]
STDaccurcy_tuple=[]
STDMspecificity_tuple=[]

#shuffle data

# 30 times
for k in range(1,11): #for 10 k nieghbors
 for p in range(1,3):  #for 2 powers
   tenFoldEvaluationq2(data,Accurcy,Sensitivity,Specificity,k,p)

   #MEANS
   Maccurcy_tuple.append([np.mean(Accurcy),k,p])#([accuracy(ypred,y_test),k,p])
   Msensitivity_tuple.append([np.mean(Sensitivity),k,p])
   Mspecificity_tuple.append([np.mean(Specificity),k,p])

   #STANDARD DEVIATIONS
   STDaccurcy_tuple.append([np.std(Accurcy),k,p])
   STDMsensitivity_tuple.append([np.std(Sensitivity),k,p])
   STDMspecificity_tuple.append([np.std(Specificity),k,p])

   Accurcy.clear()
   Sensitivity.clear()
   Specificity.clear()
  #----------------------*--------------------------------
 #sort and create dataframe, convert from tuple for Mean Acurrcy
 Maccurcy_tuple = sorted(Maccurcy_tuple, key=itemgetter(2))
 MeanaccuracyDFrame = pd.DataFrame(list(Maccurcy_tuple))
 
 #Seperate mean acurrcy by power ->list
 Meanaccuracy_p1_list= MeanaccuracyDFrame.iloc[0:10,0].values
 Meanaccuracy_p2_list= MeanaccuracyDFrame.iloc[10:20,0].values
 #       -----------------*-----------
 STDaccurcy_tuple=sorted(STDaccurcy_tuple, key=itemgetter(2))
 STDaccurcyDFrame = pd.DataFrame(list(STDaccurcy_tuple))
 STDMaccuracyp1_list= STDaccurcyDFrame.iloc[0:10,0].values
 STDMaccuracyp2_list= STDaccurcyDFrame.iloc[10:20,0].values
 #                    ----*----
 #plt.errorbar(x= xplaceholder, y=Meanaccuracyp1,yerr=STDMaccurcy)
 #plt.show()
  #----------------------*--------------------------------
 
 # Mean sensitivity sort, convert to data frame
 Msensitivity_tuple = sorted(Msensitivity_tuple, key=itemgetter(2))
 MsensitivityDframe = pd.DataFrame(list(Msensitivity_tuple))
 
 #Seperate Mean sensitivity by power ->list
 Msensitivityp1_list = MsensitivityDframe.iloc[0:10,0].values
 Msensitivityp2_list = MsensitivityDframe.iloc[10:20,0].values
 #----------------------*--------------------------------
 
 # Mean sensitivity sort, convert to data frame
 Mspecificity_tuple = sorted(Mspecificity_tuple, key=itemgetter(2))
 MspecificityDFrame = pd.DataFrame(list(Mspecificity_tuple))
 Mspecificityp1_list= MspecificityDFrame.iloc[0:10,0].values
 Mspecificityp2_list= MspecificityDFrame.iloc[10:20,0].values
 #----------------------*--------------------------------
 STDMspecificity_tuple =sorted(STDMspecificity_tuple, key=itemgetter(2))
 STDMspecificityDFrame = pd.DataFrame(list(STDMspecificity_tuple))
 STDMspecificityp1_list = STDMspecificityDFrame.iloc[0:10,0].values
 STDMspecificityp2_list= STDMspecificityDFrame.iloc[10:20,0].values

 #----------------------*--------------------------------
 STDMsensitivity_tuple= sorted(STDMsensitivity_tuple, key=itemgetter(2))
 STDMsensitivityDFrame = pd.DataFrame(list(STDMsensitivity_tuple))
 STDMsensitivityp1_list=STDMsensitivityDFrame.iloc[0:10,0].values
 STDMsensitivityp2_list=STDMsensitivityDFrame.iloc[10:20,0].values
 
 
 #----------------------*--------------------------------
 #X=

#Q2Plots(Accurcy,Sensitivity,Specificity)











#print(results)
