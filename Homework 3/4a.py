import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

if __name__ == '__main__':
	IMAGEFILE = 'lenna.jpg'
	img=mpimg.imread(IMAGEFILE)
	lenna = np.array(img, dtype=np.float64) / 255

	w, h, d = original_shape = tuple(lenna.shape)
	assert d == 3
	image_array = np.reshape(lenna, (w * h, d))

	kmeans = KMeans(n_clusters=8).fit(image_array)
	labels = kmeans.predict(image_array)

	image = np.zeros((w, h, d))
	label_idx = 0
	for i in range(w):
		for j in range(h):
			image[i][j] = image_array[labels[label_idx]]
			label_idx += 1

	fig,axs = plt.subplots(nrows=1,ncols=2)
	axs[0].imshow(lenna)
	axs[1].imshow(image)
	plt.show()
