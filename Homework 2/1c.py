import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

from logistic_regression import LogisticRegression


if __name__ == '__main__':
	TRAINFILE = 'titanic_train.csv'
	TESTFILE = 'titanic_test.csv'
	train_df = pd.read_csv(TRAINFILE,header=0)
	test_df = pd.read_csv(TESTFILE,header=0)

	x_data = train_df.iloc[:, 2:]

	# drop useless columns
	x_data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

	# one hot sex and embarked
	x_data = pd.get_dummies(x_data, columns=['Sex', 'Embarked'])

	# fill missing age data with mean....
	x_data['Age'] = x_data['Age'].fillna(x_data['Age'].mean())

	# normalize with zscores
	numeric_cols = ['Parch', 'SibSp', 'Age', 'Fare']
	x_data[numeric_cols] = x_data[numeric_cols].apply(zscore)

	x = x_data.as_matrix().astype(float)
	y = train_df.iloc[:, 1].as_matrix().astype(float)


	x_data_test = test_df.iloc[:, 2:]

	# drop useless columns
	x_data_test.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

	# one hot sex and embarked
	x_data_test = pd.get_dummies(x_data_test, columns=['Sex', 'Embarked'])

	# fill missing age data with mean....
	x_data_test['Age'] = x_data_test['Age'].fillna(x_data_test['Age'].mean())

	# normalize with zscores
	x_data_test[numeric_cols] = x_data_test[numeric_cols].apply(zscore)

	x_test = x_data_test.as_matrix().astype(float)
	y_test = test_df.iloc[:, 1].as_matrix().astype(float)

	#learning rate, tensor
	eta = np.eye(10) * np.array([0.000001] * 10)

	#number of iterations
	epochs = 100000

	#weights
	w = np.array([-1.,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])

	lr = LogisticRegression(eta=eta, epochs=epochs, w=w, enable_early_stop=True, early_stop_tolerance=10)

	error = lr.fit(x, y)
	print("training error: {}".format(error))

	test_error = lr.score(x_test, y_test)
	print("test error: {}".format(test_error))

	iterations = len(lr.cost_over_epochs)
	print("training iterations: {}".format(iterations))

	plt.plot(lr.cost_over_epochs, 'k')
	plt.plot(lr.gradiant_over_epochs, 'r')
	plt.xlabel('training iterations N')
	plt.ylabel('gradiants, cost')
	plt.show()
