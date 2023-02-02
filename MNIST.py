from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

digit = load_digits()
X = digit.data
y = digit.target


dig=pd.DataFrame(digit.data)
#pd.DataFrame(digit.target).head()
#print(dig.sample(5))
#plt.imshow(digit.images[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

print(X_train[3])
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("kNN model accuracy:",acc)


