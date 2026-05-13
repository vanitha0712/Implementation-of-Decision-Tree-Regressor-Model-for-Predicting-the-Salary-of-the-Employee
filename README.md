# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and preprocess the data Load dataset and encode categorical values
2. Split features and target Define independent variable X and dependent variable y
3.  Train the model Fit DecisionTreeRegressor to the data
4. Predict and visualize Predict new values and plot the results


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: B.vanitha
RegisterNumber: 212225220117
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/acer/Downloads/Salary.csv")

le = LabelEncoder()
data['Position'] = le.fit_transform(data['Position'])

X = data.iloc[:, 1:2].values  
y = data.iloc[:, 2].values   

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])
print(f"Predicted Salary for Level 6.5: ${y_pred[0]}")

import numpy as np
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Regression')
plt.title('Salary vs Level (Decision Tree Regressor)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

## Output:
<img width="752" height="617" alt="image" src="https://github.com/user-attachments/assets/b7c71815-03b9-45c4-8086-5236d5e1eea5" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
