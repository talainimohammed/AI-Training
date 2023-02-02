
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics



# Charger le dataset iris
iris = load_iris()

# Mettre les attributs dans la mtrice X et l’étiquette dans le vecteur y
X = iris.data
y = iris.target
print(y)
# feature_names représente les caractéristiques 
# et target_names représente les espèces (Setosa, Versicolor, Verginica).
feature_names = iris.feature_names
target_names = iris.target_names

# afficher features_names et target_names 
print("Feature names:", feature_names)
print("Target names:", target_names)
#Feature names: ['sepal length (cm)','sepal width (cm)', 'petal length (cm)','petal width (cm)']
#Target names: ['setosa' 'versicolor' 'virginica']
# afficher les 5 premières lignes de X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# afficher les dimensions
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# training the model on training set
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# faire des prédictions sur l'ensemble de test
y_pred = knn.predict(X_test)
# comparer les valeurs des étiquettes réelles (y_test) aux valeurs des étiquettes prédites (y_pred)
acc = metrics.accuracy_score(y_test, y_pred)
print("kNN model accuracy:",acc)

sample = [[0.5, 2, 5, 2], [2, 4, 8, 4]]
preds = knn.predict(sample)
pred_species = [iris.target_names[p] for p in preds]
print("Predictions:", pred_species)

    
