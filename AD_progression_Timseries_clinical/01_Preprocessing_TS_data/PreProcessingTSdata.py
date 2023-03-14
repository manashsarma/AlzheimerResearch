#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 07:57:18 2019

@author: manashsarma
"""



import pandas as pd
import numpy as np

# convert series to supervised learning
def checkVISCODEgap(currMonth, prevMonth):
	m = (currMonth - prevMonth - 6) /6     	
	return int(m)


str_exp = ''

tadpoleD1D2File = 'TADPOLE_D1_D2.csv' 

Dtadpole=pd.read_csv(tadpoleD1D2File)


Dtadpole=Dtadpole.rename(columns={'DXCHANGE':'Diagnosis'})
# Select features
Dtadpole=Dtadpole[['RID','COLPROT', 'ORIGPROT','Diagnosis','AGE', 'ADAS13','Ventricles','ICV_bl','VISCODE']].copy()  

Dtadpole['VISCODE'] = Dtadpole['VISCODE'].map({'bl': 0, 'm06': 6, 'm12': 12, 'm18': 18, 'm24': 24, 'm30': 30, 'm36': 36, 'm42': 42,\
                                        'm48': 48,'m54': 54}) 

# https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-conditions-on-column-values/
indexNames = Dtadpole[ (Dtadpole['COLPROT'] == 'ADNI2') | (Dtadpole['COLPROT'] == 'ADNIGO') ].index
Dtadpole.drop(indexNames, inplace=True)
#Dtadpole.to_csv('TADPOLE_D1_D2_Mod.csv')
#idx=Dtadpole.index

#1. Iterate each index
#2. For each rid, check how many rows has this rid,
#2.1 if only one rid, remove this rid.
# if more than one rows for same rid, maintain a gap of 6 for viscode till

#kIdenx = Dtadpole.index
#enth = len(Dtadpole)


'''
We can use numpy.insert. This has the advantage of flexibility. You only need to specify the index you want to insert to.

s1 = pd.Series([5, 6, 7])
s2 = pd.Series([7, 8, 9])

df = pd.DataFrame([list(s1), list(s2)],  columns =  ["A", "B", "C"])

pd.DataFrame(np.insert(df.values, 0, values=[2, 3, 4], axis=0))

    0   1   2
0   2   3   4
1   5   6   7
2   7   8   9

For np.insert(df.values, 0, values=[2, 3, 4], axis=0), 0 tells the function the place/index you want to place the new values.
'''

Dtadpole = Dtadpole.sort_values(['RID', 'VISCODE'])


# first remove those rows for which any field is null for the same RID.

'''rids = Dtadpole['RID']
#nRids = len(rids)

rids = rids.drop_duplicates(keep='first') 
len2 = len(rids)
for i in range(len2):'''

rids = Dtadpole['RID']
rids = rids.drop_duplicates(keep='first')
ridsV = rids.values 
len2 = len(rids)

for i in range(0,len2):
    
    ix = (Dtadpole['RID'] == ridsV[i])
    dd = Dtadpole[ix]
    
    ff = pd.notna(dd['Ventricles'])
    cc = dd['Ventricles'].isna()
    
    if ( (cc.any() == True) & (ff.any() == True)) :

         dd.loc[cc,'Ventricles'] = dd.loc[ff,'Ventricles'].values[0]
         Dtadpole[ix] = dd
    
    ix1 = (Dtadpole['RID'] == ridsV[i])
    dd1 = Dtadpole[ix1]
    ff1 = pd.notna(dd1['ADAS13'])
    cc1 = dd['ADAS13'].isna()
    
    if ( (cc1.any() == True) & (ff1.any() == True)) :

         dd1.loc[cc1,'ADAS13'] = dd1.loc[ff1,'ADAS13'].values[0]
         Dtadpole[ix1] = dd1     
    
    # In case, a particular colmn for all rows in dd is blank 
    # please insert value for nan for all columns in this set of rows
    # for column loop itertaion
    #  for loop iteration through 'dd', the selected rows for same rid
    
    # here remove the rows that has a field with nan for all column  
    if ( (dd['Ventricles'].isna().all() == True) | (dd['Diagnosis'].isna().all() == True) \
     | (dd['AGE'].isna().all() == True) | (dd['ADAS13'].isna().all() == True)\
     | (dd['Ventricles'].isna().all() == True) | (dd['ICV_bl'].isna().all() == True) \
     | (dd['VISCODE'].isna().all() == True) | ( len(dd) == 1 ) ):  
          Dtadpole = Dtadpole[Dtadpole.RID != ridsV[i]] 

prevVCODE = -1
prevRID = -1
valDtap = Dtadpole.values
lenth =  10*len(valDtap)
for i in range (0, lenth ) :
    lenth2 =  len(valDtap)
    if ( (i+1) > lenth2):
        break;
    #i = i + gap    
    vcode = valDtap[i][8]
    rID = valDtap[i][0]
    if (rID == prevRID):
         '''COLPROT = valDtap[i][1]
         if (pd.isna(COLPROT)):
             COLPROT = prevCOLPROT
         ORIGPROT = valDtap[i][2]
         if (pd.isna(ORIGPROT)):
             ORIGPROT = prevORIGPROT
         Diagnosis = valDtap[i][3]
         if (pd.isna(Diagnosis)):
             Diagnosis = prevDiagnosis
         AGE = valDtap[i][4]
         if (pd.isna(AGE)):
             AGE = prevAGE
         ADAS13 = valDtap[i][5]
         if (pd.isna(ADAS13)):
             ADAS13 = preADAS13
         Ventricles = valDtap[i][6]
         if (pd.isna(Ventricles)):
             Ventricles = prevVentricles
         ICV_bl = valDtap[i][7]
         if (pd.isna(ICV_bl)):
             ICV_bl = prevICV_bl'''
             
         gap=checkVISCODEgap(vcode, prevVCODE)
         
         if (gap > 0):
             # create new rows and insert
             #print('\n There is gap of %d for rid : %d' % (gap, rID))
             #print('\n earlier vist code:', prevVCODE)
             #print('current vist code:', vcode)
                 
             for j in range (0, gap):
                 # create a row and insert
                 prevVCODE = prevVCODE + 6
                 COLPROT = valDtap[i][1]
                 if (pd.isna(COLPROT)):
                     COLPROT = prevCOLPROT
                 ORIGPROT = valDtap[i][2]
                 if (pd.isna(ORIGPROT)):
                     ORIGPROT = prevORIGPROT
                 Diagnosis = valDtap[i][3]
                 if (pd.isna(Diagnosis)):
                     Diagnosis = prevDiagnosis
                 AGE = valDtap[i][4]
                 if (pd.isna(AGE)):
                     AGE = prevAGE
                 ADAS13 = valDtap[i][5]
                 if (pd.isna(ADAS13)):
                     ADAS13 = preADAS13
                 Ventricles = valDtap[i][6]
                 if (pd.isna(Ventricles)):
                     Ventricles = prevVentricles
                 ICV_bl = valDtap[i][7]
                 if (pd.isna(ICV_bl)):
                     ICV_bl = prevICV_bl
                 row_value = [rID, COLPROT, ORIGPROT, Diagnosis, AGE, ADAS13, Ventricles, ICV_bl, prevVCODE]
                 #print(row_value)
                 # insert the row into the array
                 valDtap=np.insert(valDtap, i+j, row_value, axis=0)
              # over - create new rows and insert    
                 
    prevRID = rID    
    prevVCODE = vcode
    prevCOLPROT = COLPROT
    prevORIGPROT = ORIGPROT
    prevAGE = AGE
    preADAS13 = ADAS13
    prevVentricles = Ventricles
    prevICV_bl = ICV_bl
    prevDiagnosis = Diagnosis
  

# Here take the missing values from previous ones.

prevVCODE = -1
prevRID = -1
#valDtap = Dtadpole.values

lenth =  10*len(valDtap)

for i in range (0, lenth ) :
    lenth2 =  len(valDtap)
    if ( (i+1) > lenth2):
        break;  
    
    rID = valDtap[i][0]    
    COLPROT = valDtap[i][1]
    ORIGPROT = valDtap[i][2]
    Diagnosis = valDtap[i][3]
    AGE = valDtap[i][4]
    ADAS13 = valDtap[i][5]
    Ventricles = valDtap[i][6]
    ICV_bl = valDtap[i][7]
    
    
    if ((pd.isna(COLPROT)) & (valDtap[i-1][0] == rID)):
         valDtap[i][1] = valDtap[i-1][1]
         
    if ((pd.isna(ORIGPROT)) & (valDtap[i-1][0] == rID)):
        valDtap[i][2] = valDtap[i-1][2]
    
    if ((pd.isna(Diagnosis)) & (valDtap[i-1][0] == rID)):
        valDtap[i][3] = valDtap[i-1][3]
    
    if ((pd.isna(AGE)) & (valDtap[i-1][0] == rID)):
        valDtap[i][4] = valDtap[i-1][4]
    
    if ((pd.isna(ADAS13)) & (valDtap[i-1][0] == rID)):
       valDtap[i][5] = valDtap[i-1][5]
    
    if ((pd.isna(Ventricles)) & (valDtap[i-1][0] == rID)):
        valDtap[i][6] = valDtap[i-1][6]
  
    if ((pd.isna(ICV_bl)) & (valDtap[i-1][0] == rID)):
        valDtap[i][7] = valDtap[i-1][7]
    
#sort_by_life = dataframe.sort_values('rID')
    
# now convert the array back to  dataframe
dataframe=pd.DataFrame(valDtap, columns=['rID', 'COLPROT', 'ORIGPROT', 'Diagnosis', 'AGE', 'ADAS13', 'Ventricles', 'ICV_bl', 'prevVCODE'])    
    
