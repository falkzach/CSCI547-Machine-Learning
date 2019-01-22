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

	poly_test_accuracies = []
	radial_test_accuracies = []

	for i in range(100):
		X,X_test,y,y_test = train_test_split(X_df,y_df,test_size=0.33)

		clf_poly = svm.SVC(kernel='poly')
		clf_radial = svm.SVC(kernel='rbf')

		clf_poly.fit(X, y)
		clf_radial.fit(X, y)

		poly_test_accuracy = clf_poly.score(X_test, y_test)
		radial_test_accuracy = clf_radial.score(X_test, y_test)

		poly_test_accuracies.append(poly_test_accuracy)
		radial_test_accuracies.append(radial_test_accuracy)

	plt.hist(poly_test_accuracies)
	plt.xlabel('Test Acuracy')
	plt.ylabel('N')
	plt.title('Polynomial Test Accuracy over 100 train/test splits')
	plt.show()

	plt.hist(radial_test_accuracies)
	plt.xlabel('Test Acuracy')
	plt.ylabel('N')
	plt.title('Radial Test Accuracy over 100 train/test splits')
	plt.show()