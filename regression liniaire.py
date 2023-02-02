# 1. Import necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/21260/Desktop/Master/Python/AI/USA_Housing.csv")

print("Dataset dimensions:", data.shape)

print("First 5 observations: \n", data.head())

print("Random sample of 10 observations: \n", data.sample(10))

print("List of columns: \n", data.columns)

sns.jointplot(x='Avg. Area Number of Rooms', y='Price', data=data)
plt.show()

X = data.drop(['Address','Price'], axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

coeffs = model.coef_
intercept = model.intercept_
print("Coeff: \n", coeffs)
print("Intercept: \n", intercept)

predictions = model.predict(X_test)
print("Predictions: \n", predictions)

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE: ", mean_absolute_error(y_test, predictions))
print("MSE: ", mean_squared_error(y_test, predictions))

new_obs = np.array([[2000, 3, 35, 55, 120]])
new_price = model.predict(new_obs)
print("nouveau observation: ", new_price)
