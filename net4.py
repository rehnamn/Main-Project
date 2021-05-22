import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

##loading dataset
feature=[]


import os
path=os.path.join(os.getcwd(),'train','train')
trainlabels=pd.read_csv(os.path.join(os.getcwd(),'trainLabels.csv','trainLabels.csv'))
print(trainlabels.head())
image=trainlabels['image']
label_name=trainlabels['level']
y=np.zeros((7,7))
label=[]
for i in range(0,len(image)):
    try:
        img_name=image[i]+'.jpeg'
        
        
        path=os.path.join(os.path.join(os.getcwd(),'train','train',img_name))
        print (path)
        full_size_image = cv2.imread(path,0)
    
                
        resized_image=cv2.resize(full_size_image, (7,7), interpolation=cv2.INTER_CUBIC)
   
                
        equ = cv2.equalizeHist(resized_image)

                
    
                
        pmin=equ.min()
        pmax=equ.max()
        s=equ.shape
                

        for r in range(0,s[0]):
            for c in range(0,s[1]):
                y[r,c]=(equ[r,c]-pmin)/(pmax-pmin)
    
                
                
        feature.append(y)
        label.append(label_name[i])
    except cv2.error: print('image not found')

t=len(feature)       
##cv2.destroyAllWindows()
feature=np.asarray(feature)
feature = feature.reshape(t,7,7,1).astype('float32')

label=np.asarray(label).astype('uint8')




from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.3)


# one hot encode outputs
from keras.utils import np_utils
label_train = np_utils.to_categorical(label_train)
label_test = np_utils.to_categorical(label_test)

num_classes = label_test.shape[1]

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
    model1.add(Conv2D(16, (3,3),input_shape=(7,7, 1), activation='relu'))
    model1.add(Conv2D(36, (3,3), activation='relu'))
##    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.2))
    model1.add(Flatten())
    model1.add(Dense(128, activation='relu'))
    model1.add(Dense(num_classes, activation='softmax'))
	
    model1.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.005), metrics=['accuracy'])
    return model1

net4=shallow_cnn1()
net4.summary()

# Fit the model
net4.fit(feature_train, label_train, validation_data=(feature_test, label_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
loss,acc = net4.evaluate(feature_test, label_test, verbose=2)
print("CNN Accuracy: " ,acc*100)

    
# serialize model to JSON
model_json = net4.to_json()
with open("net4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
net4.save_weights("net4.h5")

with open("acc_net4.json", "w") as json_file:
    json_file.write(str(acc))
