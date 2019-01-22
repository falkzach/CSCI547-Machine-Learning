from sklearn import datasets

if __name__ == "__main__":
	datasets.fetch_lfw_people(min_faces_per_person=50,resize=0.5)

	print("Done!")
