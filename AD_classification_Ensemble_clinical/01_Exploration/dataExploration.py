#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:49:59 2019

@author: manashsarma
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:47:23 2019

@author: manashsarma
"""
import pandas_profiling
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#import sys
#!{sys.executable} -m pip install pandas-profiling
#1. some important data is in text string like 'MCI'/ 'CN'/ 'Dementia'
# convert them to integer

#2. Rows with NULL value for label 'DX' variable to be removed

# Consider the file received from ETL process

data = pd.read_csv("ADNIMERGE.csv" ) 
df = data

df.head()

df.dtypes.sort_values()

null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()

pandas_profiling.ProfileReport(df)

df.loc[df['VISCODE']=='bl']

df = df.loc[df['VISCODE']=='bl']
df.shape[0]

plt.hist(df["AGE"], bins=20)
plt.xlabel('Age of Native')
plt.ylabel('No of Natives')
plt.show()

noAPOE4_MCI_0 = (df[(df.DX == 'CN') & (df.APOE4 == 0)] ).shape[0]
noAPOE4_MCI_1 = (df[(df.DX == 'CN') & (df.APOE4 == 1)] ).shape[0]
noAPOE4_MCI_2 = (df[(df.DX == 'CN') & (df.APOE4 == 2)] ).shape[0]

objects = ('APOE4 count 0', 'APOE4 count 1', 'APOE4 count 2')
y_pos = np.arange(len(objects))
performance = [noAPOE4_MCI_0,noAPOE4_MCI_1, noAPOE4_MCI_2]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of natives of healthy category')
plt.xlabel('Number of APOE4 gene allele in natives')
plt.title('Alzimer disease sample statistics')


noAPOE4_MCI_0 = (df[(df.DX == 'MCI') & (df.APOE4 == 0)] ).shape[0]
noAPOE4_MCI_1 =  (df[(df.DX == 'MCI') & (df.APOE4 == 1)] ).shape[0]
noAPOE4_MCI_2 = (df[(df.DX == 'MCI') & (df.APOE4 == 2)] ).shape[0]

objects = ('APOE4 count 0', 'APOE4 count 1', 'APOE4 count 2')
y_pos = np.arange(len(objects))
performance = [noAPOE4_MCI_0,noAPOE4_MCI_1, noAPOE4_MCI_2]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of natives with mild cognitive issue')
plt.xlabel('Number of APOE4 gene allele in natives')
plt.title('Alzimer disease sample statistics')

noAPOE4_AD_0 = (df[(df.DX == 'Dementia') & (df.APOE4 == 0)] ).shape[0]
noAPOE4_AD_1 =  (df[(df.DX == 'Dementia') & (df.APOE4 == 1)] ).shape[0]
noAPOE4_AD_2 = (df[(df.DX == 'Dementia') & (df.APOE4 == 2)] ).shape[0]

objects = ('APOE4 count 0', 'APOE4 count 1', 'APOE4 count 2')
y_pos = np.arange(len(objects))
performance = [noAPOE4_AD_0,noAPOE4_AD_1, noAPOE4_AD_2]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of natives of dementia category')
plt.xlabel('Number of APOE4 genes in natives')
plt.title('Alzimer disease sample statistics')

col_names = [ 'VISCODE','AGE', 'APOE4', 'MMSE','DX', 'SITE', 'PTMARRY', 'ADAS11', 'ADAS13', 
             'RAVLT_forgetting', 'RAVLT_immediate', 'PTRACCAT','PTGENDER','Hippocampus',''
             'PTETHCAT', 'RAVLT_learning','Ventricles', 'ICV', 'mPACCdigit','Entorhinal','Fusiform']

dataframe = df[col_names]

dataframe=dataframe.loc[dataframe['VISCODE']=='bl']

dataframe.drop('VISCODE', axis=1, inplace=True)

# Now convert  to indeger values

dataframe['PTMARRY'] = dataframe['PTMARRY'].map({'Married': 0, 'Divorced': 1, 'Widowed': 2, 'Never married': 3, 'Unknown': 4})

dataframe['PTRACCAT'] = dataframe['PTRACCAT'].map({'White': 0, 'Black': 1})

dataframe['PTGENDER'] = dataframe['PTGENDER'].map({'Female': 0, 'Male': 1})

dataframe['DX'] = dataframe['DX'].map({'CN': 0, 'MCI': 1, 'Dementia': 2})

dataframe['PTETHCAT'] = dataframe['PTETHCAT'].map({'Not Hisp/Latino': 0, 'Hisp/Latino': 1, 'Unknown': 2})


corrmat = dataframe.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(25,25))
#plot heat map
g=sns.heatmap(dataframe[top_corr_features].corr(),annot=True,cmap="RdYlGn")