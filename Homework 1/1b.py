import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#
#	degree - an arbitrary degree of polynomial
#	gamma - regularaztion strength (constant)
#	dataset - a pandas dataframe
#
def fit_with_l2(degree, gamma, dataset, test_set):
	#data and target from padas frame
	x = dataset[0].as_matrix().astype(float)
	y = dataset[1].as_matrix().astype(float)

	x_test = test_set[0].as_matrix().astype(float)
	y_test = test_set[1].as_matrix().astype(float)

	xhat = np.linspace(x.min(),x.max(), 200)
	Xhat = np.vander(xhat, N=degree + 1, increasing=True)
	x = 2*(dataset[0] - x.min())/(x.max()-x.min()) - 1  #this is x_tilde
	x_test = 2*(test_set[0] - x.min())/(x.max()-x.min()) - 1

	# Vandermond matrix (design matrix)
	X = np.vander(x,degree+1,increasing=True)

	# Identitity matrix for regularization
	Eye = np.eye(X.shape[1])
	#Eye hat, to not regularize bias
	Eye[0, 0] = 0

	# Solve for weight's set (params)  (training)
	w = np.linalg.solve(np.dot(X.T,X) + gamma*Eye,np.dot(X.T,y))

	yhat = np.dot(Xhat,w) # 250x2, 16x1
	
	X_test = np.vander(x_test, N=degree+1, increasing=True)

	avg_rmse = np.sqrt(np.sum((np.dot(X,w) - y)**2)/len(y))
	avg_rmse_test = np.sqrt(np.sum((np.dot(X_test, w) - y_test)**2)/len(y_test))

	plt.plot(x,y,'ro')
	plt.plot(xhat,yhat,'k-')
	plt.plot(x_test,test_set[1],'b*')
	plt.plot(x_test,y_test,'b^')
	plt.xlabel('Arbitrary Dataset Feature')
	plt.ylabel('Arbitrary Dataset Target')
	plt.show()

if __name__ == "__main__":
	dataset_file = "P1C_training.csv"
	testset_file = "P1C_test.csv"
	dataset = pd.read_csv(dataset_file, header=None)
	testset = pd.read_csv(testset_file, header=None)
	fit_with_l2(15, 1e-4, dataset, testset)

