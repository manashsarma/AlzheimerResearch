#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:15:29 2020

@author: manashsarma
"""

import types
import pandas as pd

df = pd.read_csv(body)

df.head()

col_names = [ 'VISCODE','AGE', 'APOE4', 'MMSE','DX', 'SITE', 'PTMARRY', 'ADAS11', 'ADAS13', 
             'RAVLT_forgetting', 'RAVLT_immediate', 'PTRACCAT','PTGENDER','Hippocampus',''
             'PTETHCAT', 'RAVLT_learning','Ventricles', 'ICV', 'mPACCdigit','Entorhinal','Fusiform']
dataframe = df[col_names]
dataframe.head()

dataframe.to_csv(file_name='ADNIMERGE_ETL.csv',overwrite=True)