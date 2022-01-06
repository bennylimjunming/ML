#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:59:01 2020

@author: bennylimjunming
"""

import pandas as pd
import numpy as np
import csv
import math
import random
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
#result = np.random.RandomState(42)
mydataset=pd.read_csv('LifeCycleSavings V1.02.csv')
#detecting which attribute have missing values
mydataset.isnull().sum()

#deleting the rows with NAN
print("The rows before deletion")
print(mydataset.shape)
print("The rows after deletion")
mydataset.dropna(inplace=True)
print(mydataset.shape)

#detect the duplicate data with the attribute country by boolean
mydataset.Country.duplicated()
#detect the total number of duplicated data
mydataset.Country.duplicated().sum()
#delete the duplication data
mydataset.drop_duplicates(keep='first', inplace=True)
print(mydataset.shape)
#fix the decimals places in the dataset to 0 so it looks neat
mydataset.round(decimals=0)

#check for outliers in the dataset.
#I assumed there is outlier in growth_rate, due to abnormal numbers compared to 95% of it)
#proven there are 2 outliers in growth_rate, which are 299 and 219.
plt.scatter(x=mydataset.index,y=mydataset['Growth_Rate'])
#outliers detected, hence it needs to be remove.
mydataset.drop([21,38], inplace=True)
print(mydataset.shape)
mydataset


#naive bayes part
#create a new dataset name mydataset, because under naive bayes test and train dataset is going to be used. 
#dataset2 act as a test set.
dataset2 =pd.read_csv('LifeCycleSavings V1.02.csv')
#First part, group income into 3 categories, which are low_income, middle_income, and high_income
#create another attribute 'income' to classify the incomes into 3 different category
def income (inc):
    if inc < 1000: return 'Low Income'
    elif 1000 < inc <= 1999: return 'Middle_Income'
    else: return 'High Income'
#using map function to call Income, so that income (which are discretize can be clasffied into 3 category based on the attribute Income)
mydataset ["income"] = mydataset['Income'].map(income)
dataset2["income"] = dataset2['Income'].map(income)
#since dataset is a test set, it is needed to delete the income and country to prevent further noise
#because i am comparing one dataset with income and one dataset which is the (test set) that does not have income to make predictions
#bascially under this part i am re-doing the cleaning process(same as at the beginning, just different dataset) 
#before i start doing naive bayes
del dataset2['Income']
del dataset2['Country']
del dataset2['income']
print("The rows before deletion")
print(dataset2.shape)
print("The rows after deletion")
dataset2.dropna(inplace=True)
print(dataset2.shape)
#drop duplicated rows because dataset2 is from the new dataset provided for the assignment
dataset2.drop([21,38], inplace=True)
print(dataset2.shape)
print(dataset2)
dataset2.drop_duplicates(keep='first', inplace=True)
print(dataset2.shape)
#delete NA row number 10
dataset2.drop([10], inplace=True)
#naive bayes algorithm is being applied
#organize the data
label_encoder = preprocessing.LabelEncoder()
mydataset['income']= label_encoder.fit_transform(mydataset['income']) 
labels = mydataset['income']
features = dataset2
#split the data into training and test set so that naive bayes can be applied
from sklearn.model_selection import train_test_split
train,test,train_labels,test_labels = train_test_split(features,labels,test_size = 0.40,random_state = 42)
#build the model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#train the model
model = gnb.fit(train, train_labels)
#prediction is being made 
preds = gnb.predict(test)
print(preds)
#accuracy score of the model, the higher the better, means more accurate
from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,preds))

        







