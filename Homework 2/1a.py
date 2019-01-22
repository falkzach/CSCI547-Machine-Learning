import numpy as np 
import pandas as pd

from logistic_regression import LogisticRegression


if __name__ == '__main__':
	DATAFILE = 'lobster_survive.dat'
	df = pd.read_csv(DATAFILE,header=0, sep=r"\s{2,}")
	x = df.iloc[:, 0].as_matrix().astype(float)
	y = df.iloc[:, 1].as_matrix().astype(float)
	x = np.vander(x,2,increasing=True)


	#learning rate, tensor
	eta = np.array([[0.000001,0],[0,0.000000001]])

	#number of iterations
	epochs = 200000

	#weights
	w = np.array([-1.,0.5])

	lr = LogisticRegression(eta=eta, epochs=epochs, w=w)
	error = lr.fit(x, y)
	print("classification error: {}".format(error))
	