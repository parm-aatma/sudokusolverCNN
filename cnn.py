import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

path=' '# here goes the MNIST dataset's csv file's path
train=pd.read_csv(path)
print(train.head())
dic={'label':[-1]*6000}
for i in range(1,29):
    for j in range(1,29):
        dic[f'{str(i)}x{str(j)}']=[0]*6000
empty=pd.DataFrame(dic)

train=pd.concat([train,empty])
print(train.tail())
train=shuffle(train)
print(train.shape)
target=train['label']
trainI=train.drop('label',axis=1)

trainI=trainI[:].values

print(trainI.shape)

target=target.to_numpy()
target=LabelEncoder().fit_transform(target)

train=[]

for row in trainI:
    row=row.reshape(28,28,1)
    train.append(row)

train=np.array(train)

train_X,test_X,train_Y,test_Y=train_test_split(train,target,test_size=0.3,random_state=42)

train_X=train_X/255.0
test_X=test_X/255.0

train_Y=to_categorical(train_Y)
test_Y=to_categorical(test_Y)

print(train_Y[0])

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(11, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=10, batch_size=200)
model.save(' ') # here goes the path where the model will be saved

