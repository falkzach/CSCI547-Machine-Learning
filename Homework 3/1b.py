import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm

import matplotlib.pyplot as plt

if __name__ == '__main__':
	DATAFILE = 'wine.data'

	df = pd.read_csv(DATAFILE,header=None)
	X_df = df.iloc[:, 1:]
	y_df = df.iloc[:, 0]

	train_accuracies = []
	test_accuracies = []

	for i in range(100):
		X,X_test,y,y_test = train_test_split(X_df,y_df,test_size=0.33)

		clf = svm.LinearSVC()

		clf.fit(X, y)

		train_accuracy = clf.score(X, y)
		test_accuracy = clf.score(X_test, y_test)

		train_accuracies.append(train_accuracy)
		test_accuracies.append(test_accuracy)

	plt.hist(test_accuracies)
	plt.xlabel('Test Acuracy')
	plt.ylabel('N')
	plt.title('Distribution of Test Accuracy over 100 train/test splits')
	plt.show()
