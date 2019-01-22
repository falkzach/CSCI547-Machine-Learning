from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from neural_network import Network

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Import the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
X,X_test,y,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

# enumerate the class 
classes = range(len(iris.target_names))

m = X.shape[0]
n = X.shape[1]
N = len(classes)

# Produce the one-hot matrix of classes
T = np.zeros((m,N))
for t,yi in zip(T,y):
    t[yi]=1

# Instantiate a neural network
# First argument: number of nodes per layer
# Second argument: Activation functions for each layer (layer 0 is always None)
# Third argument: Whether to add a bias node
# Fourth argument: standard deviation and mean of initial guess for weights
nn = Network([n,20,N],[None,'sigmoid','softmax'],[True,True,False],layer_weight_means_and_stds=[(0,0.1),(0,0.1)])

# Learning rate
eta = 0.001

# Number of iterations to complete
N_iterations = 10000

# Perform gradient descent
for i in range(N_iterations):

    # For stochastic gradient descent, take random samples of X and T

    # Run the features through the neural net (to compute a and z)
    y_pred = nn.feed_forward(X)

    # Compute the gradient
    grad_w = nn._gradient_fun(X,T) 

    # Update the neural network weight matrices
    for w,gw in zip(nn.weights,grad_w):
        w -= eta*gw

    # Print some statistics every thousandth iteration
    if i%1000==0:
        misclassified = sum(np.argmax(y_pred,axis=1)!=y.ravel())
        print ("Iteration: {0}, Objective Function Value: {1:3f}, Misclassified: {2}".format(i,nn._J_fun(X,T), misclassified))

# Predict the training data and classify
y_pred = np.argmax(nn.feed_forward(X_test),axis=1)
print ("Test data accuracy: {0:3f}".format(1-sum(y_pred!=y_test.ravel())/float(len(y_test))))

# Print a matrix of plots showing points and misfit point
fig,axs = plt.subplots(nrows=4,ncols=4)
for i in range(4):
    for j in range(4):
        if i>j:
            axs[i,j].scatter(X_test[:,i],X_test[:,j],c=y_pred)
            axs[i,j].plot(X_test[y_test!=y_pred,i],X_test[y_test!=y_pred,j],'ro',markersize=3)
            axs[i,j].set_xlabel(iris['feature_names'][i])
            axs[i,j].set_ylabel(iris['feature_names'][j])
        else: # delete redundant plots
            fig.delaxes(axs[i,j])
plt.show()



