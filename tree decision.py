from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.head()
df.describe()
df['target'].value_counts()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)


plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

new_data = np.array([[4.9, 3.6, 1.4, 0.1]])
prediction = tree.predict(new_data)
print(iris.target_names[prediction])
