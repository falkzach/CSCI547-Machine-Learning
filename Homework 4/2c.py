import pickle
import numpy as np 

from markov_models import NaiveBayesModel

DATASET_TRAINING = "genes_training.p"
DATASET_TEST = "genes_test.p"

if __name__ == "__main__":
	training = pickle.load(open(DATASET_TRAINING, "rb"))
	test = pickle.load(open(DATASET_TEST, "rb"))

	training_data = np.array(training[0])
	training_lables = np.array(training[1])
	test_data = np.array(test[0])
	test_labels = np.array(test[1])

	sequences_0 = training_data[training_lables == 0]
	sequences_1 = training_data[training_lables == 1]

	seq_0 = ''.join(str(seq) for seq in sequences_0)
	seq_1 = ''.join(str(seq) for seq in sequences_1)

	sequence_nb_model_0 = NaiveBayesModel(seq_0)
	sequence_nb_model_0.build_transition_matrices()

	sequence_nb_model_1 = NaiveBayesModel(seq_1)
	sequence_nb_model_1.build_transition_matrices()

	predictions = []

	for sequence in test_data:
		sequence = ' '.join(sequence)
		scores = []
		scores.append(sequence_nb_model_0.compute_log_likelihood(sequence))
		scores.append(sequence_nb_model_1.compute_log_likelihood(sequence))
		predictions.append(np.argmax(scores))

	total = test_labels.size
	correct = np.sum(predictions == test_labels)
	accuracy = correct/total
	print("Accuracy: {}".format(accuracy))
