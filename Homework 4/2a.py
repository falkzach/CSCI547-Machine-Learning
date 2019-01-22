import pickle
import numpy as np 

from markov_models import FirstOrderMarkovModel

DATASET_TRAINING = "genes_training.p"

def new_sequence(class_id, models):
	return models[class_id].generate_phrase()


if __name__ == "__main__":
	training = pickle.load(open(DATASET_TRAINING, "rb"))

	training_data = np.array(training[0])
	training_lables = np.array(training[1])

	training_0 = training_data[training_lables[:] == 0]
	training_1 = training_data[training_lables[:] == 1]


	sequences_0 = training_0[0]
	seq_0 = ''.join(str(seq) for seq in sequences_0)

	sequences_1 = training_1[0]
	seq_1 = ''.join(str(seq) for seq in sequences_1)

	sequence_mm_model_0 = FirstOrderMarkovModel(seq_0)
	sequence_mm_model_0.build_transition_matrices()

	sequence_mm_model_1 = FirstOrderMarkovModel(seq_1)
	sequence_mm_model_1.build_transition_matrices()

	models = [sequence_mm_model_0, sequence_mm_model_1]
	for i in range(0,2):
		print(new_sequence(i, models))
