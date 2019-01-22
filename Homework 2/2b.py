import numpy as np
import matplotlib.pyplot as plt

from neural_network import Network


if __name__ == '__main__':
	n = 1
	m = 100
	N = 1
	X = np.random.rand(m, 1)
	y = np.exp(-np.sin(np.power(X, 3) * 4 * np.pi))

	X_test = np.random.rand(m, 1)
	y_test = np.exp(-np.sin(np.power(X_test, 3) * 4 * np.pi))

	nn = Network([n,20,N],[None,'sigmoid','identity'],[True,True,False],layer_weight_means_and_stds=[(0,0.1),(0,0.1)],regularization="L2")

	eta = 0.001

	N_iterations = 10000

	T = y

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
