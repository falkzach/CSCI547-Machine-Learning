import numpy as np 
import pandas as pd


# The sigmoid function
def _sigmoid(w,X):
	z = np.dot(X,w)
	return 1./(1+np.exp(-z))


# The objective function
def _J_fun(w,X,y):
	return -sum(y*np.log(_sigmoid(w,X)) + (1-y)*np.log(1-_sigmoid(w,X)))


# The gradient of the objective function
def _gradient_fun(w,X,y):
	return np.dot(_sigmoid(w,X)-y,X)


class LogisticRegression:
	def __init__(self, eta=None, epochs=10000, w=None, enable_early_stop=False, early_stop_tolerance=10):
		self.eta= eta
		self.epochs = epochs
		self.w = w
		self.cost_over_epochs = []
		self.gradiant_over_epochs = []

		self.enable_early_stop = enable_early_stop
		self.early_stop_tolerance = early_stop_tolerance


	def fit(self, x, y):
		N = len(y)

		for i in range(self.epochs):
			grad_w = _gradient_fun(self.w,x,y)    # Compute the gradient of the objective function
			self.w -= np.dot(self.eta,grad_w)
			self.cost_over_epochs.append(_J_fun(self.w,x,y))
			self.gradiant_over_epochs.append(grad_w)

			if(self.shouldTerminateEarly()):
				break;

		classification_error = sum((_sigmoid(self.w,x)>0.5)==y)/N
		return classification_error


	def score(self, x, y):
		N = len(y)

		classification_error = sum((_sigmoid(self.w,x)>0.5)==y)/N
		return classification_error

	def shouldTerminateEarly(self):
		if(self.enable_early_stop == True):
			grads = self.gradiant_over_epochs[-1]
			return (all(g < self.early_stop_tolerance for g in grads)
				and all(g > -1 * self.early_stop_tolerance for g in grads) )
		return False
