import warnings
warnings.filterwarnings('ignore')
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np

from keras.models import model_from_json
net0 = open('net0.json', 'r')
net0_json = net0.read()
net0.close()
loaded_net0 = model_from_json(net0_json)
loaded_net0.load_weights("net0.h5")


net1 = open('net1.json', 'r')
net1_json = net1.read()
net1.close()
loaded_net1 = model_from_json(net1_json)
loaded_net1.load_weights("net1.h5")

net2 = open('net2.json', 'r')
net2_json = net2.read()
net2.close()
loaded_net2 = model_from_json(net2_json)
loaded_net2.load_weights("net2.h5")

net3 = open('net3.json', 'r')
net3_json = net3.read()
net3.close()
loaded_net3 = model_from_json(net3_json)
loaded_net3.load_weights("net3.h5")

net4 = open('net4.json', 'r')
net4_json = net4.read()
net4.close()
loaded_net4 = model_from_json(net4_json)
loaded_net4.load_weights("net4.h5")

net5 = open('net5.json', 'r')
net5_json = net5.read()
net5.close()
loaded_net5 = model_from_json(net5_json)
loaded_net5.load_weights("net5.h5")

filename=askopenfilename()
img=cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


h0=cv2.resize(img,(17,17))
h1=cv2.resize(img,(11,11))
h2=cv2.resize(img,(13,13))
h3=cv2.resize(img,(7,7))
h4=cv2.resize(img,(7,7))
h5=cv2.resize(img,(13,13))

asa0 = loaded_net0.predict(h0.reshape(-1,17,17,1))
asa1 = loaded_net1.predict(h1.reshape(-1,11,11,1))
asa2 = loaded_net2.predict(h2.reshape(-1,13,13,1))
asa3 = loaded_net3.predict(h3.reshape(-1,7,7,1))
asa4 = loaded_net4.predict(h4.reshape(-1,7,7,1))
asa5 = loaded_net5.predict(h5.reshape(-1,13,13,1))
d={0:'no_dr',
   1:'mild_dr',
   2:'moderate_dr',
   3:'severe_dr',
   4:'poliferative_dr'}
out0=np.argmax(asa0,axis=1)
out1=np.argmax(asa1,axis=1)
out2=np.argmax(asa2,axis=1)
out3=np.argmax(asa3,axis=1)
out4=np.argmax(asa4,axis=1)
out5=np.argmax(asa5,axis=1)
print(d[out0[0]])
print(d[out1[0]])
print(d[out2[0]])
print(d[out3[0]])
print(d[out4[0]])
print(d[out5[0]])
acc0 = open('acc_net0.json', 'r')
acc0_json = acc0.read()
acc0_json=float(acc0_json)
acc0.close()


acc1 = open('acc_net1.json', 'r')
acc1_json = acc1.read()
acc1_json=float(acc1_json)
acc1.close()

acc2 = open('acc_net2.json', 'r')
acc2_json = acc2.read()
acc2_json=float(acc2_json)
acc2.close()

acc3 = open('acc_net3.json', 'r')
acc3_json = acc3.read()
acc3_json=float(acc3_json)
acc3.close()

acc4 = open('acc_net4.json', 'r')
acc4_json = acc4.read()
acc4_json=float(acc4_json)
acc4.close()

acc5 = open('acc_net5.json', 'r')
acc5_json = acc5.read()
acc5_json=float(acc5_json)
acc5.close()
overall_acc=acc0_json+acc1_json+acc2_json+acc3_json+acc4_json+acc5_json
A0=acc0_json/overall_acc
A1=acc1_json/overall_acc
A2=acc2_json/overall_acc
A3=acc3_json/overall_acc
A4=acc4_json/overall_acc
A5=acc5_json/overall_acc


C=(A0*asa0 + A1*asa1 + A2*asa2 + A3*asa3 + A4*asa4+ A5*asa5)/6
out_C=np.argmax(C,axis=1)
print('integrated')
print(d[out_C[0]])
