import tensorflow as tf
import numpy as np
import os
import cv2

t=400*5
path1=os.getcwd()
path=os.path.join(path1,'data')
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt		 

data=os.listdir(path)
kernel = np.ones((2,1),np.uint8)

out=np.zeros((t,227,227))
label=[]
from keras.models import load_model

d=0
k=0
m=0
for i in data:
    path1=os.path.join(path,i)
    print(path1)
    class_data=os.listdir(path1)
    print(class_data)
    m=d
    d=d+1
    
    da=[m for x in range(len(class_data))]
    label.extend(da)
    m=m+1
    for j in class_data:
        
        
        imag=cv2.imread(os.path.join(path1,j),0)
        imag=cv2.resize(imag,(227,227))

        ou = cv2.adaptiveThreshold(imag,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)

        ou = cv2.morphologyEx(ou, cv2.MORPH_OPEN, kernel)
        ou= ou.reshape(227,227)
        cv2.imshow('eye',ou)
        cv2.waitKey(1)
        ou=255-ou
        ou=np.multiply(imag,ou)
        cv2.imshow('eye',ou)
        cv2.destroyAllWindows
        cv2.waitKey(1)
        out[k,:,:]=ou
        
        k=k+1
cv2.destroyAllWindows()


from keras import datasets, layers, models




out=out/255
u=out.reshape(t,227,227,1)
print(label)

x_train, x_valid, y_train, y_valid = train_test_split(u, label, test_size=0.10, shuffle= True)
model = models.Sequential()
model.add(layers.Conv2D(64, (11, 11), activation='relu', input_shape=(227, 227, 1)))
model.add(layers.MaxPooling2D((2, 2),strides=(2,2)))
model.add(layers.Conv2D(96, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2),strides=(2,2)))
model.add(layers.Conv2D(256, (4, 4), activation='relu'))
model.add(layers.MaxPooling2D((2, 2),strides=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(50))
model.add(layers.Dense(50))
model.add(layers.Dense(5, activation='softmax'))

model.summary()

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, verbose=2)
loss, acc = model.evaluate(x_valid,y_valid, verbose=2,batch_size=24)
model.save('modelcnn1_num.h5')
print("Accuracy is:",acc)
