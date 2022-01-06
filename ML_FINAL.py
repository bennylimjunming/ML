#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:58:16 2020

@author: bennylimjunming
"""

import pandas as pd
import numpy as np
import csv
import math
import random
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from IPython import get_ipython
from keras.models import Sequential
from keras.layers import Dense
get_ipython().run_line_magic('matplotlib', 'inline')

#import the dataset
ds= pd.read_csv('forest_fires_dataset V1.03.csv')

#detect missing values on which attribute
print("The result before mean")
print(ds)
#replace all the attributes with mean
ds['temp'].fillna((ds['temp'].mean()), inplace=True)
ds['humidity'].fillna((ds['humidity'].mean()), inplace=True)
ds['wind'].fillna((ds['wind'].mean()), inplace=True)
#print the result after it is replaced with means
print("The result after mean")
print(ds)


#check for duplicate data
ds.duplicated()
#display the total rows of duplicated data 
ds.duplicated().sum()
#delete the duplicated row 
ds.drop_duplicates(keep='first',inplace=True)
print(ds.shape)

#round the decimals places. 
ds=ds.round(decimals=0)
print(ds)
#detect the outliers from the dataset
plt.scatter(x=ds.index,y=ds['burned_area'])
plt.scatter(x=ds.index,y=ds['rain'])
#delete the outliers
ds.drop([22,42],inplace=True)
print(ds.shape)


#Question 3, Naive Bayes is being used
#load the dataset that has been cleaned from above into name ds1
ds1=ds
#use the value of burned_area provided, 
#to make prediction whether there is fire or no,
ds1.loc[ds1.burned_area != 0, 'burned_area'] = "ON FIRE"
ds1.loc[ds1.burned_area == 0, 'burned_area'] = "NO FIRE"
print(ds1)
#load ds1 into ds2
ds2= ds1
print(ds2)
#divide the x and y columns by using iloc to get rows and columns at specific position
X = ds2.iloc[:,0:3]
y = ds2.iloc[:,3]
#split the dataset into train and test set, 30% is used for test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#build the model 
model = GaussianNB()
#fit the training set into the model
model.fit(X_train,y_train)
#predict the model score using test set
model.score(X_test,y_test)
#the first 15 rows of the data is being selected for the test set
X_test[0:15]
y_test[0:15]
#using x_test to make predictions for the model
model.predict(X_test[0:15])
model.predict_proba(X_test[:15])
#cross validation is being executed 
from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(),X_train, y_train, cv=5)








