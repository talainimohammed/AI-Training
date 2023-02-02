from tkinter import *
import re
from tkinter.messagebox import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=3)
iris = load_iris()

def Model():
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)
    #knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("kNN model accuracy:",acc,"estt:0.25")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=44)
    #knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("kNN model accuracy:",acc,"test:0.35")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=42)
    #knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("kNN model accuracy:",acc,"test:0.45")

def Prediction():
    sample = [[int(sl.get()),int(sw.get()),int(pl.get()),int(pw.get())]]
    preds = knn.predict(sample)
    pred_species = [iris.target_names[p] for p in preds]
    #print("Predictions:", pred_species)
    showinfo("Predictions",pred_species)

ws = Tk()
ws.title('IRIS')
ws.geometry('500x400')
ws.config(bg="#447c84")

frame = Frame(ws, padx=20, pady=20)
Label(frame,text="IRIS",font=("Times", "24", "bold")).grid(row=0, columnspan=3, pady=10)

Label(frame, text='Sepal length:', font=("Times", "14")).grid(row=1, column=0, pady=5)


Label(frame, text='Sepal width:', font=("Times", "14") ).grid(row=2, column=0, pady=5)

Label(frame, text='petal length:', font=("Times", "14")).grid(row=3, column=0, pady=5)

Label(frame, text='petal width:', font=("Times", "14")).grid(row=4, column=0, pady=5)


sl = Entry(frame, width=30)
sl.grid(row=1, column=1)

sw = Entry(frame, width=30)
pl = Entry(frame, width=30)
pw = Entry(frame, width=30)

sw.grid(row=2, column=1)
pl.grid(row=3, column=1)
pw.grid(row=4, column=1)

# button
model = Button(frame, text="Model", padx=20, pady=10, relief=SOLID, font=("Times", "14", "bold"),command=Model )
predection = Button(frame, text="predection", padx=20, pady=10, relief=SOLID, font=("Times", "14", "bold"),command=Prediction )
model.grid(row=6, column=0, pady=20)
predection.grid(row=6, column=1, pady=20)
frame.pack(expand=True)
ws.mainloop()


