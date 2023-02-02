import pandas as pd
from tkinter import *
import re
from tkinter.messagebox import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=5)

notes=pd.read_csv("dataset_notes.csv")

def Model():
    x = notes.iloc[:, :4].values
    y = notes['DECISION'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=1)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("kNN model accuracy:",acc,"estt:0.2")
  

def Prediction():
    sample = [[int(sl.get()),int(sw.get()),int(pl.get()),int(pw.get())]]
    preds = knn.predict(sample)
    msg=""
    if(preds==0):
        msg="Non validé"
    else:
        msg="validé"
    showinfo("Decision:", msg)

ws = Tk()
ws.title('Notes')
ws.geometry('500x400')
ws.config(bg="#447c84")

frame = Frame(ws, padx=20, pady=20)
Label(frame,text="Notes",font=("Times", "24", "bold")).grid(row=0, columnspan=3, pady=10)

Label(frame, text='Python:', font=("Times", "14")).grid(row=1, column=0, pady=5)


Label(frame, text='JAVA:', font=("Times", "14") ).grid(row=2, column=0, pady=5)

Label(frame, text='PHP:', font=("Times", "14")).grid(row=3, column=0, pady=5)

Label(frame, text='MSproject:', font=("Times", "14")).grid(row=4, column=0, pady=5)


sl = Entry(frame, width=30)
sl.grid(row=1, column=1)

sw = Entry(frame, width=30)
pl = Entry(frame, width=30)
pw = Entry(frame, width=30)

sw.grid(row=2, column=1)
pl.grid(row=3, column=1)
pw.grid(row=4, column=1)

# button
model = Button(frame, text="Generer le Model", padx=20, pady=10, relief=SOLID, font=("Times", "14", "bold"),command=Model )
predection = Button(frame, text="Afficher La Decision", padx=20, pady=10, relief=SOLID, font=("Times", "14", "bold"),command=Prediction )
model.grid(row=6, column=0, pady=20)
predection.grid(row=7, column=0, pady=20)
frame.pack(expand=True)
ws.mainloop()


