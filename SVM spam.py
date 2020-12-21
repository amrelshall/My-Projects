#imports
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn import svm

#---------------------------------------------------------
# functions
def plotData(X, y ,S):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    
    plt.scatter(X[pos,0], X[pos,1], s=S, c='b', marker='+', linewidths=1)
    plt.scatter(X[neg,0], X[neg,1], s=S, c='r', marker='o', linewidths=1)
    

def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plotData(X, y,6)
    #plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='y', marker='|', s=100, linewidths='5')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)
    
#--------------------------------------------------------- 

# Training
spam_train = loadmat('C://Users//Amr Elshall//ml-coursera-python-assignments-master//Exercise6//Data//spamTrain.mat')
spam_test = loadmat('C://Users//Amr Elshall//ml-coursera-python-assignments-master//Exercise6//Data//spamTest.mat')

print('spam train')
print(spam_train)
print('spam test')
print(spam_test)

# select X ,Xtest , y , ytest
X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

# print shapes
print(X.shape, y.shape, Xtest.shape, ytest.shape)

svc = svm.SVC()
svc.fit(X, y)

# Testing and accuracy
accuracy = format(np.round(svc.score(Xtest, ytest) * 100, 2))
print('Test accuracy = ' , accuracy , '%')
