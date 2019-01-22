import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
	digits = load_digits()
	data = np.round(digits['data']/16.0)

	X = data    # n x m matrix of features 
	y = digits.target  # n vector of classes
	X,X_test,y,y_test = train_test_split(X,y,test_size=0.33,random_state=42) # Split into 33% test and 67% training sets 

	classes = np.linspace(0,9,10) 

	m = X.shape[0]  # Number of data instances
	m_test = X_test.shape[0] # Number of test data instances
	N = 10           # Number of classes
	n = X.shape[1]  # Number of features

	mu_array = np.zeros((n,N))
	sigma2_array = np.zeros((n,N))
	prior_array = np.zeros((N))

	#Learning phase
	for k in range(N):    #Loop over each class label
	    C_k = classes[k]
	    prior = sum(y==C_k)/float(y.shape[0])                           # Count the number of data where the label is C_k
	    mu = np.sum(X[y==C_k],axis=0)/len(X[y==C_k])                    # Take the mean of those features where the corresponding label is C_k
	    mu_array[:,k] = mu                                              # Store in the arrays we created above
	    prior_array[k] = prior

	class_probabilities = np.zeros((m,N))  # The probabilities for 

	for i,x in enumerate(X):  # Loop over the training data instances
	    for k in range(N):    # Loop over the classes
	        prior = prior_array[k]
	        mu = mu_array[:,k]
	        # likelihood = np.prod(np.exp(-(x-mu)**2/(2*sigma2))) #change me
	        likelihood = np.prod(np.power(mu, x) * np.power((1-mu), 1-x) )
	        posterior_k = prior*likelihood
	        class_probabilities[i,k] = posterior_k
	        
	class_probabilities /= np.sum(class_probabilities,axis=1,keepdims=True)

	y_pred_train = np.argmax(class_probabilities,axis=1)

	cm_train = confusion_matrix(y,y_pred_train)
	print(cm_train)
	print("training accuracy:", 1 - sum(abs(y!=y_pred_train))/float(m))


	# Test set predictions
	class_probabilities = np.zeros((m_test,N))

	for i,x in enumerate(X_test):
	    for k in range(N):
	        prior = prior_array[k]
	        mu = mu_array[:,k]
	        sigma2 = sigma2_array[:,k]
	        likelihood = np.prod(np.power(mu, x) * np.power((1-mu), 1-x) )
	        posterior_k = prior*likelihood
	        class_probabilities[i,k] = posterior_k

	class_probabilities /= class_probabilities.sum(axis=1,keepdims=True)
	y_pred_test = np.argmax(class_probabilities,axis=1)

	cm_test = confusion_matrix(y_test,y_pred_test)
	print(cm_test)
	print("test accuracy:", 1 - sum(abs(y_test!=y_pred_test))/float(m_test))

