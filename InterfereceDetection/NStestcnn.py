# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import keras
from tensorflow.keras.utils import to_categorical
import numpy as np

from os.path import dirname, join as pjoin
import scipy.io as sio
print(tf.version.VERSION)

set1 = sio.loadmat('setnsfft.mat')

labels = set1['FFTLabels']

data=np.zeros((len(labels),set1['trainFFT1'].shape[0],set1['trainFFT1'].shape[1],1))

for x in range (1,101):
 name = 'trainFFT' + str(x)
 TM2D = set1[name]
 TM3D = TM2D.reshape((TM2D.shape[0], TM2D.shape[1], 1))
 data[x-1,:,:] = TM3D




data1 = data
print(np.amax(data))
#data1 [data!=0] = np.log10(data1[data!=0])
print(np.amax(data1))
data_norm=data/np.amax(data)
print(np.amax(data_norm))

import pandas as pd
C= str(np.unique(labels))
print(C)

s = pd.Series(['0', '1', '2', '4', '8', '16'], dtype="category")
print(s.cat.categories)

labels [labels == 0] = 0
#labels [labels == 1] = 1
#labels [labels == 2] = 2
#labels [labels == 4] = 3
#labels [labels == 8] = 4
labels [labels == 16] = 1

C= str(np.unique(labels))
print(C)
onehot= to_categorical(labels, num_classes=len(np.unique(labels)))

np.random.seed(100)

indices = np.random.permutation(data.shape[0])

valid_cnt = int(data.shape[0] * 0.2)

#test_cnt = int(data.shape[0] * 0.1)

#valid_idx, test_idx, training_idx = indices[:valid_cnt],\
#                         indices[valid_cnt:valid_cnt+test_cnt],indices[valid_cnt+test_cnt:]

valid_idx, training_idx = indices[:valid_cnt],\
                         indices[valid_cnt:]
  
#valid, test, train = data[valid_idx,:],\
#              data[test_idx,:],data[training_idx,:]

valid, train = data[valid_idx,:],\
              data[training_idx,:]
  
#onehot_valid, onehot_test, onehot_train = onehot[valid_idx,:],onehot[test_idx,:],\
#                        onehot[training_idx,:]

onehot_valid, onehot_train = onehot[valid_idx,:],\
                        onehot[training_idx,:]

#train=train.reshape([-1,train.shape[1],train.shape[2],1])
#test=test.reshape([-1,test.shape[1],test.shape[2],1])

print('Training Data Shape=', train.shape , '\nValid Data Shape=', valid.shape)

from keras.models import Sequential

from keras.layers import MaxPooling2D, Dropout, Dense, Flatten

from keras.layers import Convolution2D as Conv2D

model = Sequential()

# 
model.add(Conv2D(8, (1, 1), activation='relu', padding = 'valid', input_shape=(2, 1000,1)))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Conv2D(4, (1, 1), activation='relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size=(1, 1)))
#model.add(Conv2D(4, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
#model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.summary()


from tensorflow.keras.optimizers import SGD


model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=0.01,  momentum=0.9, nesterov=True),
              metrics=['accuracy'])

history=model.fit(train, onehot_train,
         epochs=15,
         batch_size=10,
         validation_data=(valid, onehot_valid),
         verbose=1)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],'bo')
plt.plot(history.history['val_accuracy'],'rX')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()
# summarize history for loss
plt.plot(history.history['loss'],'bo')
plt.plot(history.history['val_loss'],'rX')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()

model.save('InterferenceFFT.h5')