# Regression Template

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the training and test set
# Remove the comment when the dataset contains sufficient data
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_tesy = train_test_split(X, y, test_size=0.2, random_state=0)
'''

# Feature scaling
# Remove the comment when requiring feature scaling
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
'''

# Method 1: Linear Regression
# Fitting the linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predicting a new result with linear regression
y_pred = lin_reg.predict([[]])

# Visualizing the linear regression test results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, lin_reg.predict(X_train), color='blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()

# Method 2: Polynomial Regression
# Fitting the polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)

# Predicting a new result with polynomial regression
y_pred = lin_reg.predict(X_poly.fit_transform([[]]))

# Visualizing the polynomial regression test results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, lin_reg.predict(poly_reg.fit_transform(X_train)), color='blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()

# Visualizing the polynomial regression test results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color='red')
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()

