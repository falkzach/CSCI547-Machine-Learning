import numpy as np 
import pandas as pd

from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt

if __name__ == '__main__':
	lfw_people = datasets.fetch_lfw_people(min_faces_per_person=50, resize=0.5)

	X = lfw_people.data
	y = lfw_people.target

	pca1 = PCA(n_components=1,copy=True,whiten=False)
	X1 = pca1.fit_transform(X)

	pca10 = PCA(n_components=10,copy=True,whiten=False)
	X10 = pca10.fit_transform(X)

	pca100 = PCA(n_components=100,copy=True,whiten=False)
	X100 = pca100.fit_transform(X)

	X1_reconstructed = 0
	for c,l in zip(pca1.components_,X1[42]):
		X1_reconstructed += c*l
	X1_reconstructed += pca1.mean_
	X1_reconstructed = X1_reconstructed.reshape((62,47))

	X10_reconstructed = 0
	for c,l in zip(pca1.components_,X10[42]):
		X10_reconstructed += c*l
	X10_reconstructed += pca10.mean_
	X10_reconstructed = X10_reconstructed.reshape((62,47))

	X100_reconstructed = 0
	for c,l in zip(pca1.components_,X100[42]):
		X100_reconstructed += c*l
	X100_reconstructed += pca100.mean_
	X100_reconstructed = X100_reconstructed.reshape((62,47))

	fig,axs = plt.subplots(nrows=1,ncols=3)
	axs[0].imshow(X1_reconstructed)
	axs[1].imshow(X10_reconstructed)
	axs[2].imshow(X100_reconstructed)
	plt.show()
