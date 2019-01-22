import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm

if __name__ == '__main__':
	DATAFILE = 'wine.data'

	df = pd.read_csv(DATAFILE,header=None)
	X = df.iloc[:, 1:]
	y = df.iloc[:, 0]

	X,X_test,y,y_test = train_test_split(X,y,test_size=0.33)

	clf = svm.LinearSVC()

	clf.fit(X, y)

	train_accuracy = clf.score(X, y)
	test_accuracy = clf.score(X_test, y_test)

	print("training accuracy: {}".format(train_accuracy))
	print("test accuracy: {}".format(test_accuracy))
