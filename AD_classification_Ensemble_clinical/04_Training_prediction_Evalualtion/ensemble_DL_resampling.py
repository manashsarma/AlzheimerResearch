#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:13:34 2022

@author: manashsarma
"""
# random-splits mlp ensemble on blobs dataset
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
import keras
from IPython.display import clear_output
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.layers import Dropout
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
#from sklearn.model_selection import cross_val_score
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from keras.regularizers import l2
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
#from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import itertools
import pickle
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# evaluate a single mlp model
def evaluate_model(X_train, Y_train, testX, testy):
  
  # encode targets
  trainy_enc = keras.utils.to_categorical(Y_train)
  testy_enc = keras.utils.to_categorical(testy)
  
  # define model

  model = Sequential()
  
  model.add(Dense(20, input_dim=8, kernel_initializer='normal',  
  activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
  model.add(Dropout(0.2))
  
  model.add(Dense(20, kernel_initializer='normal', activation='relu', 
  kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
  model.add(Dropout(0.2))
  
  model.add(Dense(3, kernel_initializer='normal', activation='softmax', 
  kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))) 

  # Compile model
  
  adamopt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
  epsilon=1e-08, decay=0, amsgrad=False)  
  model.compile(loss='categorical_crossentropy', optimizer = adamopt , metrics=['accuracy'])
  
  
  # Convert labels to categorical one-hot encoding
  one_hot_labels = keras.utils.to_categorical(Y_train, num_classes=3)
  one_hot_labelsTs = keras.utils.to_categorical(testy, num_classes=3)              

  # Fit the model
  
  weights = {0:117, 1:100, 2:263}
  
  es = EarlyStopping(monitor='val_accuracy', min_delta=0.001, mode='max', verbose=1, patience=200)
  mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1,
    save_best_only=True)
  
  history = model.fit(X_train, one_hot_labels, batch_size=5, validation_data=(testX, one_hot_labelsTs),
                      epochs=4000, verbose=1, callbacks=[es, mc])
  
  # load the saved model
  saved_model = load_model('best_model.h5')
  # evaluate the model
  _, test_acc = saved_model.evaluate(testX, testy_enc, verbose=0)
  return history, model, test_acc


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
  # make predictions
  yhats = [model.predict(testX) for model in members]
  yhats = array(yhats)
  # sum across ensemble members
  summed = np.sum(yhats, axis=0)
  # argmax across classes
  result = argmax(summed, axis=1)
  return result

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
  # select a subset of members
  subset = members[:n_members]
  # make prediction
  yhat = ensemble_predictions(subset, testX)
  # calculate accuracy
  return accuracy_score(testy, yhat)

# make an ensemble prediction for multi-class classification
def ensemble_predictions_for(members, n_members, testX):
  # make predictions
  # yhats = [model.predict(testX) for model in members]
  # yhats = array(yhats)
  # # sum across ensemble members
  # summed = numpy.sum(yhats, axis=0)
  # # argmax across classes
  # result = argmax(summed, axis=1)
  
  # select a subset of members
  subset = members[:n_members]
  # make prediction
  result = ensemble_predictions(subset, testX)
  
  return result

def plot_confusion_matrix(Y_test, t):
    cmap=None
    normalize=False
    cm = confusion_matrix(Y_test, t)
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    target_names = ['CN', 'MCI', 'AD']#[0, 1, 2]
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    
    if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")   
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average) 


  
df=pd.read_csv('ADNIMERGE_processed.csv')

df.head()

label = df['DX']
df.drop('DX', axis=1, inplace=True)

X, y = df, label

# Split into ensemble model construction and validation data
# trainX / trainy for model construction
# testX / testy for validation 
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.25)

# multiple train-test splits
n_splits = 25

historyDLlist, scores, members = list(), list(), list()

for _ in range(n_splits):
# split data
  trX, tsX, trY, tsY = train_test_split(trainX, trainy, test_size=0.25)
  # evaluate model

  historyDL, model, test_acc = evaluate_model(trX, trY, tsX, tsY) 
  print('>%.3f' % test_acc)
  scores.append(test_acc)
  members.append(model)
  historyDLlist.append(historyDL)

# summarize expected performance
print('Estimated Accuracy %.3f (%.3f)' % (mean(scores), std(scores)))

# make prediction
yhat = ensemble_predictions(members, testX)

print ("\n\nClassification report SVC classifier : \n", classification_report(testy, yhat))
print("Accuracy:",accuracy_score(testy, yhat))
plot_confusion_matrix(testy, yhat)

#############


# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, n_splits+1):
   ensemble_score = evaluate_n_members(members, i, testX, testy)
   newy_enc = to_categorical(testy)
   _, single_score = members[i-1].evaluate(testX, newy_enc, verbose=0)
   print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score)) 
   ensemble_scores.append(ensemble_score)
   single_scores.append(single_score)

# plot score vs number of ensemble members
print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))

x_axis = [i for i in range(1, n_splits+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None') 
pyplot.plot(x_axis, ensemble_scores, marker='o')
pyplot.show()

# make prediction with all numbers
yhat = ensemble_predictions(members, testX)

print ("\n\nClassification report  classifier : \n", classification_report(testy, yhat))
print("Accuracy:",accuracy_score(testy, yhat))
plot_confusion_matrix(testy, yhat)

# confusion matrix for max and roc auc
yhat2 = ensemble_predictions_for(members, 15, testX)
print ("\n\nClassification report classifier : \n", classification_report(testy, yhat2))
print("Accuracy:",accuracy_score(testy, yhat2))
plot_confusion_matrix(testy, yhat2)

roc_auc = multiclass_roc_auc_score(testy, yhat2 )

####### roc curve ############

n_classes = 3

y1 = label_binarize(testy, classes=[0, 1, 2])
y_score = label_binarize(yhat2, classes=[0, 1, 2])
y_test = y1

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])    
    
# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

target_names = ['CN', 'MCI', 'AD']
# Plot linewidth.
lw = 2

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(target_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
############

#model training
def ploTrainingLoss(histries):
  # make predictions
  #yhats = [model.predict(testX) for model in members]
  subset = histries[:4]
  i = 1
  for history in subset:
     print(history.history.keys())
     plt.subplot(2,2,i)
     # summarize history for loss
     plt.plot(history.history[ 'loss' ])
     #plt.plot(history.history[ 'val_loss'])
     plt.title( 'model loss' )
     plt.ylabel( 'loss' )
     plt.xlabel( 'epoch' )
     plt.legend([ 'train' ], loc= 'upper left' )
     plt.show()
     i = i+1

def ploTrainingAccuracy(histries):
  # make predictions
  #yhats = [model.predict(testX) for model in members]
  subset = histries[:4]
  i = 1
  for history in subset:
     print(history.history.keys())
     plt.subplot(2,2,i)
     # summarize history for accuracy
     #plt.plot(history.history[ 'accuracy' ])
     plt.plot(history.history[ 'val_accuracy' ])
     plt.title( 'model test accuracy' )
     plt.ylabel( 'accuracy' )
     plt.xlabel( 'epoch' )
     #plt.legend([ 'train' ,  'test' ], loc= 'upper left' )
     plt.legend([ 'test' ], loc= 'upper left' )
     plt.show()
     i = i+1

ploTrainingAccuracy(historyDLlist)     
ploTrainingLoss(historyDLlist)

#yhats = [model.predict(testX) for score in scores]:
for score in scores:    
    print("score:\n",score)

############### 

# from matplotlib import pyplot as plt
# import numpy as np
#  
#  
# # Creating dataset
# cars = ['white', 'Alaskan', 'Asian','Black', 'Hawaiian', 'Unknown / Other']
#         
#  
# data = [662, 2, 10, 29, 2, 8]
#  
# # Creating plot
# fig = plt.figure(figsize =(10, 7))
# plt.pie(data, labels = cars)
#  
# # show plot
# plt.show()