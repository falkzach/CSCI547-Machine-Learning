import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def fit_with_l2(degree, gamma, dataset, test_set):
	#data and target from padas frame
	x = dataset[0].as_matrix().astype(float)
	y = dataset[1].as_matrix().astype(float)

	x_test = test_set[0].as_matrix().astype(float)
	y_test = test_set[1].as_matrix().astype(float)

	xhat = np.linspace(x.min(),x.max(), 200)
	Xhat = np.vander(xhat, N=degree + 1, increasing=True)
	x = 2*(dataset[0] - x.min())/(x.max()-x.min()) - 1  #this is x_tilde
	x_test = 2*(test_set[0] - test_set[0].min())/(test_set[0].max()-test_set[0].min()) - 1

	# Vandermond matrix (design matrix)
	X = np.polynomial.legendre.legvander(x,degree)

	# Identitity matrix for regularization
	Eye = np.eye(X.shape[1])
	#Eye hat, to not regularize bias
	Eye[0, 0] = 0

	# Solve for weight's set (params)  (training)
	w = np.linalg.solve(np.dot(X.T,X) + gamma*Eye,np.dot(X.T,y))

	yhat = np.dot(Xhat,w) # 250x2, 16x1
	
	X_test = np.polynomial.legendre.legvander(x_test, degree)

	avg_rmse = np.sqrt(np.sum((np.dot(X,w) - y)**2)/len(y))
	avg_rmse_test = np.sqrt(np.sum((np.dot(X_test, w) - y_test)**2)/len(y_test))
	return avg_rmse, avg_rmse_test

if __name__ == "__main__":
	dataset_file = "P1C_training.csv"
	testset_file = "P1C_test.csv"
	dataset = pd.read_csv(dataset_file, header=None)
	testset = pd.read_csv(testset_file, header=None)

	degs = []
	rmses = []
	rmses_test = []

	for i in range(1, 21):
		avg_rmse, avg_rmse_test = fit_with_l2(i, 0, dataset, testset)
		degs.append(i)
		rmses.append(avg_rmse)
		rmses_test.append(avg_rmse_test)

	plt.plot(degs,rmses,'b-')
	plt.plot(degs, rmses_test, 'r-')
	plt.xlabel('Degree N')
	plt.ylabel('Average RMSE')
	plt.show()
