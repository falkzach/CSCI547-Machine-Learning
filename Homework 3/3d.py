import numpy as np 
import pandas as pd

from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
	lfw_people = datasets.fetch_lfw_people(min_faces_per_person=50, resize=0.5)

	X = lfw_people.data
	y = lfw_people.target

	pca = PCA(n_components=100,copy=True,whiten=True)
	X = pca.fit_transform(X)

	print (np.cumsum(pca.explained_variance_ratio_))

	X,X_test,y,y_test = train_test_split(X,y,test_size=0.33)

	clf = svm.LinearSVC()

	y_pred = clf.fit(X, y).predict(X_test)

	train_accuracy = clf.score(X, y)
	test_accuracy = clf.score(X_test, y_test)

	print("training accuracy: {}".format(train_accuracy))
	print("test accuracy: {}".format(test_accuracy))

	cnf_matrix = confusion_matrix(y_test, y_pred)
	print(cnf_matrix)
