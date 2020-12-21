#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data 
path = 'D:\Programing\ML\SK Learn\Machine Learning A-Z Template Folder\Part 2 - Regression\P14-Part2-Regression\Section 6 - Simple Linear Regression\Python\Salary_Data.csv'
dataset = pd.read_csv(path)
print(dataset)

#turning data to matrix
col = dataset.shape[1]
X = dataset.iloc[ : , :col-1]
y = dataset.iloc[ : , col-1:col]
X = np.array(X)
y = np.array(y)

#splittig the dataset to traning set and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.25 , random_state = 0)

#fiting simple linear regression to training set
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train , y_train)

#predicting the test set
y_pred = linear_regression.predict(X_test)

#ploting the training set
plt.scatter(X_train , y_train , color = 'blue')
plt.plot(X_train , linear_regression.predict(X_train) , color = 'red')
plt.title('salary vs YearsExperience(training set)')
plt.xlabel('YearsExperience')
plt.ylabel('salary')

#ploting the testing set
plt.scatter(X_test , y_test , color = 'blue')
plt.plot(X_train , linear_regression.predict(X_train) , color = 'red')
plt.title('salary vs YearsExperience(testing set)')
plt.xlabel('YearsExperience')
plt.ylabel('salary')
plt.show()