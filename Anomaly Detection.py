# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
  
# GD function  //Calculate average(mu) and sigma  to applying the law
def GD(X):
    mu = X.mean(axis=0) #  mean 
    sigma = X.var(axis=0) # variance 
    
    return mu, sigma

# select the value of threshold 
def select_threshold(p, y):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    
    step = (p.max() - p.min()) / 1000
    
    for epsilon in np.arange(p.min(), p.max(), step):
        preds = p < epsilon
        
        # True positive , false Positive , false negative
        tp = np.sum(np.logical_and(preds == 1, y == 1))
        fp = np.sum(np.logical_and(preds == 1, y == 0))
        fn = np.sum(np.logical_and(preds == 0, y == 1))
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    
    return best_epsilon, best_f1

#----------------------------------------------------------------------------
    
# load data
data = loadmat('C://Users//Amr Elshall//ml-coursera-python-assignments-master//Exercise8//Data//ex8data1.mat')
print(data)
X = data['Xval']
y = data['yval']
print(X.shape)
print(y.shape)


# ploting data # and hist 
plt.scatter(X[ : , 0] , X[ : , 1] , s = 40 , c = 'b' , edgecolors='black' ,marker = 'o')
plt.hist(X , histtype='bar')

# applaying GD function to get the mean(mu) and variance(sigma) to applying the low
mu, sigma = GD(X)
print('mu = ' , mu)
print('sigma = ' , sigma)

dist = stats.norm(mu[0], sigma[0])
dist.pdf(X[:,0])[0:X.shape[0]]


p = np.zeros((X.shape[0], X.shape[1]))
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])
print(p)

# get the epsilon value and f1 score for the accuracy
epsilon, f1 = select_threshold(p, y)
print('epsilon = ' , epsilon)
print('F1 score = ' , f1)

# find anomaly points if p < epsilon
anomaly_point = np.where(p < epsilon)

# select anomaly points in a graph 
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[anomaly_point[0],0], X[anomaly_point[0],1], s=50, color='r', marker='o')
