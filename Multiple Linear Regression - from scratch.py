# calling libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cost function 
def computeCost(X, y, theta):
    z = np.power(((X * theta.T) - y), 2) # inner_multiplication in the law 
    J = np.sum(z) / (2 * len(X)) # divition by 1/2m
    return J

# gradient descent function
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

# -----------------------------------------------------------

# read data
path = 'C://Users//Amr Elshall//ml-coursera-python-assignments-master//Exercise1//Data//ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size' , 'Bedrooms' , 'Price'])

## Data Preprocessing

# show data
print('data = ')
print(data2) # print data
print()
print('data.describe = ')
print(data2.describe()) # print data description

# rescaling data to make the data between (-1) to (1)
data2 = (data2 - data2.mean()) / data2.std()  # division by (std) or (range)
print()
print('data after rescaling = ')
print(data2)

# add ones column ( X0 )
data2.insert(0, 'Ones', 1)

# separate X (training data) from y (target variable)
cols = data2.shape[1]
X = data2.iloc[:,0:cols-1]
y = data2.iloc[:,cols-1:cols]


# convert to matrices and initialize theta
X = np.matrix(X)
y = np.matrix(y)

# defult value for theta
theta = np.matrix([0,0,0]) #I need 3 theates theta0 , 1 , 2 all = 0 (defult) 

print('X = \n',X)
print('X.shape = ' , X.shape)
print('---------------------------------')
print('y = \n', y)
print('y.shape = ' , y.shape)
print('---------------------------------')
print('theta = \n',theta)
print('theta.shape = ' , theta.shape)

# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100

# perform linear regression on the data set
new_thetas , cost = gradientDescent(X, y, theta, alpha, iters)

# get the cost (error) of the model
J = computeCost(X, y, new_thetas)


print('new thetas = ' , new_thetas)
print('cost  = ' , cost[0:10] ) # the first 10 valuse
print('computeCost = ' , J) # the lowest value in cost (J)
print('-------------------------------')

# get best fit line for Size vs. Price
x = np.linspace(data2.Size.min(), data2.Size.max(), 100) #split the size to 100 point or more
print('x \n',x)

f = new_thetas[0, 0] + (new_thetas[0, 1] * x) # h(x)
print('f \n',f)

# draw the line for Size vs. Price    relation between x(the 100 point for size) and f(the linear line h(x))
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data2.Size, data2.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')


# get best fit line for Bedrooms vs. Price
x = np.linspace(data2.Bedrooms.min(), data2.Bedrooms.max(), 100)
print('x \n',x)


f = new_thetas[0, 0] + (new_thetas[0, 1] * x)
print('f \n',f)

# draw the line  for Bedrooms vs. Price    relation between x(the 100 point for bedrooms) and f(the linear line h(x))
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data2.Bedrooms, data2.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

# draw error graph  relation between J(cost value) and iters(the number of iterations)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
