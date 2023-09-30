#Q1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Q2
df=pd.read_csv("C:/Users/21260/Desktop/Master/Python/AI/control ai/diabetes.csv")

#Q3
print(df.shape)

#Q4
print(df.head())

#Q5
print(df.dtypes)

#Q6
print("La Moyenne")
print(df.mean())
print("La Max")
print(df.max())
print("La Min")
print(df.min())

#Q7
print(df.head(5))

#Q8
print(df.tail(5))

#Q9
print(df.sample(10))

#Q10
print(df.loc[:,['Outcome','BMI','Age']])

#Q11
print(df[(df.Glucose > 90) & (df.Glucose < 130)])

#Q12

print(df[(df.Outcome ==1) & (df.Pregnancies >0)])

#Q13
print(len(df[df.BloodPressure > 70]))

#Q14

print(max(df.BMI[df.Outcome==0]))

#Q15

print(df.isnull())

#Q16

plt.scatter(df.BloodPressure[df.Outcome==0], df.Glucose[df.Outcome==0] ,c='blue')
plt.scatter(df.BloodPressure[df.Outcome==1], df.Glucose[df.Outcome==1] ,c='red')
plt.xlabel("BloodPressure")
plt.ylabel("Glucose")
#plt.show()

#Q17

plt.figure(figsize=(12,10))
plt.hist(df.Age,bins=20,align="mid",rwidth=0.9,color="b")
plt.title("Frequence Par Age")
plt.xlabel("Age")
plt.ylabel("Fréquence")
plt.legend()
#plt.show()

#Q18
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Q19

#print(X_test)
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X_train, y_train)
#y_pred=clf.predict([X_test])
#score=metrics.accuracy_score(y_test, y_pred)
#print("son score est :",score)

#Q20
#tree.plot_tree(clf, filled=True)

#Q21
# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("kNN model accuracy:",acc)

#SVM
svm=SVC(probability=True)

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("svm model accuracy:",acc)

#Q22
knn_prob=knn.predict_proba(X_test)
knn_prob1=knn_prob[:,1]
fpr,tpr,thresh=metrics.roc_curve(y_test,knn_prob1)
roc_auc_knn=metrics.auc(fpr,tpr)
#################
svm_prob=svm.predict_proba(X_test)
svm_prob1=svm_prob[:,1]
fpr_svm,tpr_svm,thresh_svm=metrics.roc_curve(y_test,svm_prob1)
roc_auc_svm=metrics.auc(fpr_svm,tpr_svm)
plt.figure(dpi=80)
plt.title("KNN vs SVM")
plt.plot(fpr,tpr,'b', label='KNN Score=%0.2f'%roc_auc_knn)
plt.plot(fpr_svm,tpr_svm,'red', label='SVM Score=%0.2f'%roc_auc_svm)
plt.legend()
plt.show()


















