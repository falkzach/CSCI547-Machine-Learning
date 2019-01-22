import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# class LogisticRegression:
# 	def __init__(self, train_df, test_df = None):
# 		self.train_df = train_df
# 		self.test_df = test_df
# 		self.x = self.df.iloc[:, 0].as_matrix().astype(float)
# 		self.y = self.df.iloc[:, 1].as_matrix().astype(float)
# 		self.N = len(self.y)
# 		self.X = np.vander(x,2,increasing=True)

# 		if(test_df):
# 			self.x_test = self.test_df.iloc[:, 0].as_matrix().astype(float)
# 			self.y_test = self.test_df.iloc[:, 1].as_matrix().astype(float)
# 			self.N_test = len(self.y_test)

# The sigmoid function
def _sigmoid(w,X):
    z = np.dot(X,w)
    return 1./(1+np.exp(-z))


# The objective function
def _J_fun(w,X):
    return -sum(y*np.log(_sigmoid(w,X)) + (1-y)*np.log(1-_sigmoid(w,X)))


# The gradient of the objective function
def _gradient_fun(w,X,y):
    return np.dot(_sigmoid(w,X)-y,X)


def logistic_regression(df):
	x = df.iloc[:, 0].as_matrix().astype(float)
	y = df.iloc[:, 1].as_matrix().astype(float)
	N = len(y)

	#design matrix
	X = np.vander(x,2,increasing=True)

	#learning rate, tensor
	eta = np.array([[0.000001,0],[0,0.000000001]])

	N_iterations = 200000

	w = np.array([-1.,0.5])

	# Do N_iterations rounds of gradient descent
	for i in range(N_iterations):
	    grad_w = _gradient_fun(w,X,y)    # Compute the gradient of the objective function
	    w -= np.dot(eta,grad_w)

	classification_error = sum((_sigmoid(w,X)>0.5)==y)/len(y)
	print(classification_error)


if __name__ == '__main__':
	DATAFILE = 'lobster_survive.dat'
	data = pd.read_csv(DATAFILE,header=0, sep=r"\s{2,}")
	logistic_regression(data)
