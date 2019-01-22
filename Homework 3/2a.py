import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
	DATAFILE1 = 'pca1.npy'
	DATAFILE2 = 'pca2.npy'

	data1 = np.load(DATAFILE1)
	data2 = np.load(DATAFILE2)

	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection='3d')
	x,y,z = data1.T
	ax1.scatter(x,y,z)
	plt.show()

	fig = plt.figure()
	ax2 = fig.add_subplot(111, projection='3d')
	x,y,z = data2.T
	ax2.scatter(x,y,z)
	plt.show()