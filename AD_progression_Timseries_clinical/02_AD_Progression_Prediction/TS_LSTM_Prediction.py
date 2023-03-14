#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:27:11 2019

@author: manashsarma
"""
import pandas as pd
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        #print(names)
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

Dtadpole=pd.read_csv('TAD_POLE_RNN2.csv')
Dtadpole=Dtadpole[['Diagnosis','AGE', 'ADAS13','Ventricles','ICV_bl', 'rID']].copy()

Dtadpole['Diagnosis'] = Dtadpole['Diagnosis'].map({4: 2, 5: 3, 6: 3, 7: 1, 8: 2, 1: 1, 2:2, 3:3})

#Dtadpole['Diagnosis'] = Dtadpole['Diagnosis'].map({4: 2, 5: 3, 6: 3, 7: 1, 8: 2})

rids = Dtadpole['rID']
rids = rids.drop_duplicates(keep='first')
ridsV = rids.values 
len2 = len(rids)

columns = ['var1(t-1)','var2(t-1)', 'var3(t-1)','var4(t-1)', 'var5(t-1)', 'var1(t)', 'var2(t)', 'var3(t)' , 'var4(t)', 'var5(t)']
df22 = pd.DataFrame(columns=columns)


for i in range(0,len2):
    
    ix = (Dtadpole['rID'] == ridsV[i])
    dd = Dtadpole[ix]
    bb = dd.drop(['rID'],axis=1)
    values = bb.values
    reframed = series_to_supervised(values, 1, 1)
    df22 = [df22, reframed]
    df22 = pd.concat(df22, ignore_index= True)

#print(df22)    
    
df22.drop(reframed.columns[[6,7,8,9]], axis=1, inplace=True)
values = df22.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)

nTotal = values.shape[0]

nTrain = int ((values.shape[0] * 0.7))
nTest = nTotal - nTrain
 
train = values[:nTrain, :]
test = values[nTrain:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
# Timestep train_X.shape[1] which is 1 and dimension train_X.shape[2] which is 5
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]))) 
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
#inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1) Manash
inv_yhat = concatenate((yhat, test_X[:, 0:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_y, test_X[:, 1:]), axis=1) Manash
inv_y = concatenate((test_y, test_X[:, 0:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# Now Confusion Matrix

# forecast
inv_yhat=np.around(inv_yhat)
inv_yhat=inv_yhat.astype(int)

# actual
inv_y = inv_y.astype(int)

#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix

title='Confusion matrix'
cm = confusion_matrix(inv_y, inv_yhat)

accuracy = np.trace(cm) / float(np.sum(cm))
misclass = 1 - accuracy

classification_report(inv_y, inv_yhat)