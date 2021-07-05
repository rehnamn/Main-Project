from tkinter.filedialog import askopenfilename
from mymodule import mainfun
from tkinter import *
from tkinter import messagebox


def fun():
    result = mainfun(askopenfilename())
    messagebox.showinfo(title='Completed', message=f'Result : {result}')

window = Tk()
window.geometry('750x390')
window.title('Di')

bg = PhotoImage(file="sample.png")

label1 = Label(window, image=bg)
label1.place(x=0, y=0)

btn = Button(window,bg='blue',activebackground='green', font=('Arial', 15), text="Select an image",fg='white', height=1, width=20, command=fun)
btn.place(x=421, y=180)

window.mainloop()