import warnings

warnings.filterwarnings("ignore")
import keras
# import tensorflow as tf
import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename


def fun():
    # Tk().withdraw()
    fileName = askopenfilename()
    if fileName == '':
        print('No file selected')
        print('Program Completed')
        exit()

    kernel = np.ones((2, 1), np.uint8)

    imag = cv2.imread(fileName, 0)
    img1 = cv2.imread(fileName)
    imag = cv2.resize(imag, (227, 227))
    img1 = cv2.resize(img1, (333, 227))
    cv2.imshow('Selected image', img1)
    cv2.waitKey(4000)

    ou = cv2.adaptiveThreshold(imag, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 5)

    ou = cv2.morphologyEx(ou, cv2.MORPH_OPEN, kernel)
    ou = ou.reshape(227, 227)

    ou = 255 - ou
    ou = np.multiply(imag, ou)

    cv2.destroyAllWindows

    model = keras.models.load_model('modelcnn1_num.h5')

    OUT = model.predict(ou.reshape(-1, 227, 227, 1))

    X = np.argmax(OUT, axis=1)

    d = {0: 'no_dr',
         1: 'mild_dr',
         2: 'moderate_dr',
         3: 'severe_dr', 4: 'proliferative_dr'}
    messagebox.showinfo(title='Completed', message=f'Result : {d[X[0]]}')
    # print(d[X[0]])
    cv2.destroyAllWindows()


window = Tk()
window.geometry('750x390')
window.title('Diabetic retinopathy')
window.resizable(height=False, width=False)

bg = PhotoImage(file="sample.png")

label1 = Label(window, image=bg)
label1.place(x=0, y=0)

btn = Button(window,bg='blue',activebackground='green', font=('Arial', 15), text="Select an image",fg='white', height=1, width=20, command=fun)
btn.place(x=421, y=180)

window.mainloop()