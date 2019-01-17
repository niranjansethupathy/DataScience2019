# 2018 Niranjan Nagarajan - A Simple Linear Model to calculate expected salary based on years of experience
#
# Main libraries required for a linear model
import numpy as np
import matplotlib.pyplot as grph
import pandas as ds


# Initializing the data values to train the model
DSet = ds.read_csv('Salary_Data.csv')
Experience = DSet.iloc[:, :-1].values
Salary = DSet.iloc[:, 1].values


# Segregating the training and test DataSet values
from sklearn.cross_validation import train_test_split
Experience_train, Experience_test, Salary_train, Salary_test = train_test_split(Experience, Salary, test_size = 1/3, random_state = 0)



# For the training set the model is initialized first
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Experience_train, Salary_train)

# Using the previous initialized model to assess the future values
salary_pred = regressor.predict(Experience_test)

# Depicting the graph based on the training set
grph.scatter(Experience_train, Salary_train, color = 'orange')
grph.plot(Experience_train, regressor.predict(Experience_train), color = 'green')
grph.title('Experience versus Wages - Training')
grph.xlabel('No of Years')
grph.ylabel('Wages')
grph.show()

# Depicting the graph based on the test set values
grph.scatter(Experience_test, Salary_test, color = 'orange')
grph.plot(Experience_train, regressor.predict(Experience_train), color = 'green')
grph.title('Experience versus Wages - Testing')
grph.xlabel('No of Years')
grph.ylabel('Wages')
grph.show()