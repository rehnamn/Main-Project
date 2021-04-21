import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

import warnings
warnings.filterwarnings("ignore")

##loading dataset
feature=[]
label=[]
image_count=0
t=0
import os
path=r'C:\\project\\MainProject\\data1'
y=np.zeros((256,256))
for (root,dirs,files) in os.walk(path):
    if files !=[]:
        l=len(files)
        
        
        print(image_count)
        for i in range(0,l):
            t=t+1
            path=os.path.join(root,files[i])
            label.append(image_count)
            
            full_size_image = cv2.imread(path,0)
            cv2.imshow('full_size_image',full_size_image)
            cv2.waitKey(100)
            
            resized_image=cv2.resize(full_size_image, (256,256), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('image',resized_image)
            cv2.waitKey(100)
            
            equ = cv2.equalizeHist(resized_image)

            
            cv2.imshow('histogram equalized',equ)
            cv2.waitKey(100)
            
            pmin=equ.min()
            pmax=equ.max()
            s=equ.shape
            

            for r in range(0,s[0]):
                for c in range(0,s[1]):
                    y[r,c]=(equ[r,c]-pmin)/(pmax-pmin)
            cv2.imshow('normalised',y)
            cv2.waitKey(100)
            
            
            feature.append(y)

        image_count=image_count+1

feature=np.asarray(feature)
feature = feature.reshape(t,256,256,1).astype('float32')

label=np.asarray(label).astype('uint8')


image_size=feature.shape
s=[]
for i in range(10):
    s.append(random.randrange(0,image_size[0]))
print(s)


m=1
for images in(s):

   
    image = np.asarray(feature[images]).squeeze()

    plt.subplot(2,5,m)
    plt.imshow(image)
    plt.title(label[images])
    
    m=m+1

plt.show()
plt.close()


from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.3)


# one hot encode outputs
from keras.utils import np_utils
label_train = np_utils.to_categorical(label_train)
label_test = np_utils.to_categorical(label_test)

num_classes = 2

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
def shallow_cnn1():
	

    model1 = Sequential()
    model1.add(Conv2D(16, (3,3),input_shape=(256,256, 1), activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.2))
    model1.add(Flatten())
    model1.add(Dense(128, activation='relu'))
    model1.add(Dense(num_classes, activation='softmax'))
	
    model1.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.005), metrics=['accuracy'])
    return model1


def shallow_cnn2():
	

    model2 = Sequential()
    model2.add(Conv2D(32, (5, 5),input_shape=(28, 28, 1), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.2))
    model2.add(Flatten())
    model2.add(Dense(128, activation='relu'))
    model2.add(Dense(num_classes, activation='softmax'))
	
    model2.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.005), metrics=['accuracy'])
    return model2


model1 = shallow_cnn1()
model2 = shallow_cnn2()


model1.summary()
model2.summary()
