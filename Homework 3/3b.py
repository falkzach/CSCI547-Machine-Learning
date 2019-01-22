import numpy as np 
import pandas as pd

from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt

if __name__ == '__main__':
	lfw_people = datasets.fetch_lfw_people(min_faces_per_person=50, resize=0.5)

	X = lfw_people.data
	y = lfw_people.target

	pca = PCA(n_components=100,copy=True,whiten=False)
	pca.fit(X)
	X = pca.transform(X)

	print (np.cumsum(pca.explained_variance_ratio_))

	fig,axs = plt.subplots(nrows=2,ncols=5)
	counter = 0
	for r in axs:
		for ax in r:
			ax.imshow(pca.components_[counter,:].reshape((62,47)))
			counter+=1
	plt.show()
