#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 15:33:57 2022

@author: manashsarma
"""
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# define dataset
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
#    random_state=7)


df=pd.read_csv('ADNIMERGE_processed.csv')

df.head()

label = df['DX']
df.drop('DX', axis=1, inplace=True)

X, y = df, label
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.2,random_state=0)

# define the model
model = XGBClassifier()

# define the evaluation method

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# evaluate the model on the dataset

n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1) # report performance
#print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))