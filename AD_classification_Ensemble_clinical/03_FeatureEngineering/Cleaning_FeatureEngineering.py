#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:58:11 2020

@author: manashsarma
"""

import pandas as pd
import numpy as np
import warnings
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0
warnings.filterwarnings("ignore")

#dataframe = pd.read_csv(project.get_file('ADNIMERGE_ETL.csv'))
dataframe = pd.read_csv('ADNIMERGE_ETL.csv')

#Since our aim is to ONLY classify / categorise the natives in Alzeimer disease diagnosis, we consider
# biomarker samples collected at first/baseline visit ONLY

dataframe=dataframe.loc[dataframe['VISCODE']=='bl']

# Now drop the viscode column as we no longer will deal with it.

dataframe.drop('VISCODE', axis=1, inplace=True)

# Now convert  to indeger values

dataframe['PTMARRY'] = dataframe['PTMARRY'].map({'Married': 0, 'Divorced': 1, 'Widowed': 2, 'Never married': 3, 'Unknown': 4})

dataframe['PTRACCAT'] = dataframe['PTRACCAT'].map({'White': 0, 'Black': 1})

dataframe['PTGENDER'] = dataframe['PTGENDER'].map({'Female': 0, 'Male': 1})

dataframe['DX'] = dataframe['DX'].map({'CN': 0, 'MCI': 1, 'Dementia': 2})

dataframe['PTETHCAT'] = dataframe['PTETHCAT'].map({'Not Hisp/Latino': 0, 'Hisp/Latino': 1, 'Unknown': 2})

dataframe['PTMARRY'].map({'Married': 0, 'Divorced': 1, 'Widowed': 2, 'Never married': 3, 'Unknown': 4})

# Now drop NA / blank from 'DX' / category column, I do not want to take risk filling missing label with some arbitration.

dataframe = dataframe.dropna(axis=0, subset=['DX'])

for column in dataframe:
    if ( (column == 'PTMARRY' )| (column == 'PTRACCAT' )| (column == 'PTGENDER') | (column == 'DX') | (column == 'PTETHCAT') | (column == 'APOE4')):
      
      dataframe[column].fillna(dataframe[column].value_counts().idxmax(), inplace=True)
    else:
      dataframe[column].fillna(dataframe[column].mean(), inplace=True)
      

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler

lr = LinearRegression()

y = dataframe['DX']

dataframe.drop('DX', axis=1, inplace=True)

dataframe[:] = StandardScaler().fit_transform(dataframe)

X = dataframe

sffs = SFS(lr, 
          k_features=(1,X.shape[1]), 
          forward=False, # Backward
          floating=True, # Floating
          scoring='neg_mean_squared_error',
          cv=5)

sffs = sffs.fit(X.as_matrix(), y.as_matrix())

#sffs = sffs.fit(X, y)

a=sffs.get_metric_dict()
n=[]
o=[]
# Compute the mean cross validation score
for i in np.arange(1,X.shape[1]):
    n.append(-np.mean(a[i]['cv_scores'])) 
   
    
    
m=np.arange(1,X.shape[1])
fig4=plt.plot(m,n)
fig4=plt.title('SFBS: Mean CV Scores vs No of features')
fig4.figure.savefig('fig4.png', bbox_inches='tight')

print(pd.DataFrame.from_dict(sffs.get_metric_dict(confidence_interval=0.90)).T)

# Get the index of the minimum CV score
idx = np.argmin(n)
print ("No of selected features=",idx)
#Get the features indices for the best backward floating fit and convert to list
b=list(a[idx]['feature_idx'])
print(b)

print("#################################################################################")
# Index the column names. 
# Features from forward fit
print("Features selected in backward floating fit")
print(X.columns[b])

col_names = ['AGE', 'APOE4', 'MMSE', 'SITE', 'ADAS11', 'ADAS13', 'PTRACCAT','mPACCdigit']
newDF = dataframe[col_names]

newDF['DX'] = y
       
# Save dataframe as csv file to storage
newDF.to_csv(file_name='ADNIMERGE_processed.csv',overwrite=True)
      