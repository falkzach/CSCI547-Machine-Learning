import numpy as np 
import pandas as pd

from sklearn.decomposition import PCA

if __name__ == '__main__':
	DATAFILE1 = 'pca1.npy'
	DATAFILE2 = 'pca2.npy'

	data1 = np.load(DATAFILE1)
	data2 = np.load(DATAFILE2)

	pca1 = PCA()
	pca1.fit(data1)

	pca2 = PCA()
	pca2.fit(data2)

	print("for {}:".format(DATAFILE1))
	print (np.cumsum(pca1.explained_variance_ratio_))

	print("for {}:".format(DATAFILE2))
	print (np.cumsum(pca2.explained_variance_ratio_))
	