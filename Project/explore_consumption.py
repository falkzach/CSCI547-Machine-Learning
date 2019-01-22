import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import ParameterGrid

import tensorflow as tf

# https://arxiv.org/abs/1506.06297
# https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29#

DATASET = "drug_consumption.data"
COLUMNS = [
			"ID",
			"Age",
			"Gender",
			"Education",
			"Country",
			"Ethnicity",
			"Nscore",		# NEO-FFI-R Neuroticism
			"Escore",		# NEO-FFI-R Extraversion
			"Oscore",		# NEO-FFI-R Openness to experience
			"Ascore",		# NEO-FFI-R Agreeableness
			"Cscore",		# NEO-FFI-R Conscientiousness
			"Impulsive",	# impulsiveness measured by BIS-11
			"SS",			# sensation seeing measured by ImpSS
			"Alcohol", 		# alcohol consumption
			"Amphetamine",	# amphetamines consumption
			"Amyl",			# amyl nitrite consumption
			"Benzo",		# benzodiazepine consumption
			"Caffine",		# caffeine consumption
			"Cannabis",		# cannabis consumption, #blazeit
			"Chocolate",	# chocolate consumption
			"Cocaine",		# cocaine consumption
			"Crack",		# crack cocaine consumption
			"Ecstasy",		# ecstasy consumption
			"Heroin",		# heroin consumption
			"Ketamine",		# ketamine consumption
			"LegalH",		# legal highs consumption
			"LSD",			# LSD consumption
			"Methadone",	# methadone
			"Mushrooms",	# magic mushrooms consumption
			"Nicotine",		# nicotine consumption
			"Semer",		# fictitious drug Semeron consumption
			"VSA",			# volatile substance abuse consumption, glue sniffing
			]

			# CL0 Never Used
			# CL1 Used over a Decade Ago
			# CL2 Used in Last Decade
			# CL3 Used in Last Year
			# CL4 Used in Last Month
			# CL5 Used in Last Week
			# CL6 Used in Last Day
DRUGS = [
			"Alcohol", 		# alcohol consumption
			"Amphetamine",	# amphetamines consumption
			"Amyl",			# amyl nitrite consumption
			"Benzo",		# benzodiazepine consumption
			"Caffine",		# caffeine consumption
			"Cannabis",		# cannabis consumption, #blazeit
			"Chocolate",	# chocolate consumption
			"Cocaine",		# cocaine consumption
			"Crack",		# crack cocaine consumption
			"Ecstasy",		# ecstasy consumption
			"Heroin",		# heroin consumption
			"Ketamine",		# ketamine consumption
			"LegalH",		# legal highs consumption
			"LSD",			# LSD consumption
			"Methadone",	# methadone
			"Mushrooms",	# magic mushrooms consumption
			"Nicotine",		# nicotine consumption
			"Semer",		# fictitious drug Semeron consumption
			"VSA",			#
		]

GROUPED_DRUGS = [
			"Semer",		# fictitious drug Semeron consumption
			"Chocolate",	# chocolate consumption
			"Caffine",		# caffeine consumption
			"Nicotine",		# nicotine consumption
			"Alcohol", 		# alcohol consumption
			"Cannabis",		# cannabis consumption, #blazeit
			"Mushrooms",	# magic mushrooms consumption
			"LegalH",		# legal highs consumption
			"Ecstasy",		# ecstasy consumption
			"LSD",			# LSD consumption
			"Ketamine",		# ketamine consumption
			"Amyl",			# amyl nitrite consumption
			"Benzo",		# benzodiazepine consumption
			"Cocaine",		# cocaine consumption
			"Crack",		# crack cocaine consumption
			"Heroin",		# heroin consumption
			"VSA",			#		
			"Amphetamine",	# amphetamines consumption
			"Methadone",	# methadone
]

HEROIN_CORRELATION = [
			"Age",
			"Gender",
			"Education",
			"Country",
			"Ethnicity",
			"Nscore",		# NEO-FFI-R Neuroticism
			"Escore",		# NEO-FFI-R Extraversion
			"Oscore",		# NEO-FFI-R Openness to experience
			"Ascore",		# NEO-FFI-R Agreeableness
			"Cscore",		# NEO-FFI-R Conscientiousness
			"Impulsive",	# impulsiveness measured by BIS-11
			"SS",	
			"Heroin",	
]

# • The heroin pleiad includes crack, cocaine, methadone, and heroin;
# • The ecstasy pleiad consists of amphetamines, cannabis, cocaine, ketamine, LSD,
# magic mushrooms, legal highs, and ecstasy;
# • The benzodiazepines pleiad contains methadone, amphetamines, cocaine, and
# benzodiazepines.


LOW_RISK_DRUGS = [
			"Chocolate",	# chocolate consumption
			"Caffine",		# caffeine consumption
			"Nicotine",		# nicotine consumption
			"Alcohol", 		# alcohol consumption
			"Cannabis",		# cannabis consumption, #blazeit
]


#TODO: try this
def convert_to_binary_classification(value):
	if(value == 0 or value == 1):
		return 0
	else:
		return 1


# convert integer classes into risk log scaled risk values
def drug_class_to_value(value):
	result = 0 
	if(value == 0): # Never
		return result
	elif(value == 1): # Over a Decade
		return result
		# result = 1/(365 * 30) # should i consider the age?
	elif(value == 2):
		result = 1/(365 * 10)
	elif(value == 3):
		result = 1/(365)
	elif(value==4):
		result = 1/30
	elif(value==5):
		result = 1/7
	elif(value==6): # Last Day
		result = 1.0  - np.finfo(np.float64).min #TODO: using 1 results in NaN's, anything else breaks correlation
	return -1/np.log(result) # this maintains correlation


# a function to plot correlation
def plot_corr(df,size=10):
    corr = df.corr()
    mask = np.tri(corr.shape[0], k=-1)
    masked = np.ma.array(corr, mask=mask)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(masked, cmap='viridis')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


# wrapper method for getting datasets for tensorflow dnn
def get_inputs_for_data(X, y, target):
    x = tf.constant(X.as_matrix(), dtype=tf.float32)
    y = tf.constant(y.values, dtype=tf.float32)
    return x, y


# wrapper method for getting datasets for tensorflow validation
def get_inputs_for_validation(X_test, y_test, target):
    x = {"x": tf.constant(X_test.as_matrix(), dtype=tf.float32)}
    y = {"Y": tf.constant(y_test.values, dtype=tf.float32)}
    return x, y


# run dnn model
def run_dnn(X,X_test,y,y_test, target, config):

    ### DNN Reg
    columns = X.columns
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(columns))]
    

    ### validation monitor
    validation_metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
        "precision":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_precision,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
        "recall":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_recall,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
        "rmse":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_root_mean_squared_error,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
    }

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=lambda: get_inputs_for_validation(X, y, target),
        # x={"x": df.drop(target, 1).values},
        # y=df[[target]].values,
        every_n_steps=50,
        eval_steps=1,
        metrics=validation_metrics,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=200)


    # Build 3 layer DNN
    nn_estimator = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                 hidden_units=config['hidden_units'],
                                                 activation_fn=tf.nn.relu,
                                                 dropout=config['dropout'],
                                                 # optimizer="Adam"
                                                 optimizer=tf.train.ProximalAdagradOptimizer(
                                                    learning_rate=config['learning_rate'],
                                                    l1_regularization_strength=0.001
                                                  ),
    )

    # Fit the deep model.
    fit = nn_estimator.fit(
        input_fn=lambda: get_inputs_for_data(X, y, target),
        steps=config['epochs'],
		# monitors=[validation_monitor]
    )

    print("DNN Regressor Fit: {}".format(fit))

    # Evaluate of the modelaccuracy.
    evaluation = nn_estimator.evaluate(
        input_fn=lambda: get_inputs_for_data(X_test, y_test, target),
        steps=1,
        metrics={}
    )

    print("DNN Regressor Evaluation: {}".format(evaluation))
    return evaluation, nn_estimator


# init a fresh csv file
def init_csv(filename, columns):
    output_df = pd.DataFrame(columns=columns)
    output_df.to_csv(filename, index=False)


# append experiment results to a csv file
def write_csv_line(filename, columns, entry):
    output_df = pd.DataFrame([entry], columns=columns)
    with open(filename, 'a') as f:
        output_df.to_csv(f, header=False, index=False)


def dnn_sk_parameter_search(X,X_test,y,y_test, target, config):

	output_csv_filename = './parameter_search_dnn.csv'
	output_columns = ['algorithm', 'target', 'metric', 'score', 'parameters', 'accuracy']
	init_csv(output_csv_filename, output_columns)

	param_sets = list(ParameterGrid(config))


	for param_set in param_sets:
		evaluation, nn_estimator = run_dnn(X,X_test,y,y_test, TARGET_DRUG, param_set)
		entry = {
			'algorithm': "tf dnn classifier",
			'target': target,
			'metric': "loss",
			'score': evaluation['loss'],
			'parameters': param_set,
			'accuracy': evaluation['accuracy']
		}
		write_csv_line(output_csv_filename, output_columns, entry)


if __name__ == "__main__":
	# CONFIGS

	doBinary = False

	doPLOTS = False
	doPCA = False
	doKLUSTER = False

	doSVM = False
	doRandomForest = False
	doDecisionTree = False
	doNerualNetwork = True
 
	TARGET_DRUG = "VSA"
	print("Running for target drug: {}".format(TARGET_DRUG))

	#
	# Read and Preprocess
	#
	df = pd.read_csv(DATASET, sep=',', header=None, names=COLUMNS)
	df = df.drop(["ID"], axis=1) # drop redundent ID column
	df = df.replace(regex=True,to_replace=r"CL",value=r"").apply(pd.to_numeric) # make targets numeric
	
	# Scale drug responses to be used as features
	drug_df = df[DRUGS]
	drug_df = drug_df[GROUPED_DRUGS] # reorder 
	drug_df_scaled = drug_df.applymap(drug_class_to_value) # map classes to values
	# print(df.describe(include='all'))
	# print(drug_df.describe(include='all'))
	# print(drug_df_scaled.describe(include='all'))
	if(doPLOTS):
		plot_corr(drug_df_scaled)
		plot_corr(df)
		plot_corr(df[HEROIN_CORRELATION])


	#
	# Train Test Split
	#
	X_df = df.drop(DRUGS, axis=1)
	if(doBinary):
		drug_df.applymap(convert_to_binary_classification)
		print("Doing binary classification")
	y_df = drug_df[TARGET_DRUG]
	drug_df_scaled = drug_df_scaled.drop(TARGET_DRUG, axis=1)
	
	# low_risk_as_pred_df = drug_df_scaled[LOW_RISK_DRUGS]
	# X_df = pd.concat([X_df, low_risk_as_pred_df], axis=1)

	X,X_test,y,y_test = train_test_split(X_df,y_df,test_size=0.33)

	#
	# Exploratory data analysis
	#
	if(doPCA):
		pca = PCA(n_components=len(X_df.columns))
		pca.fit(X_df)
		print("PCA for {}:".format(DATASET))

		print("Described Variance by Compontent")
		print(np.cumsum(pca.explained_variance_ratio_))
		print(pd.DataFrame(pca.components_,columns=X_df.columns,index=X_df.columns))

	if(doKLUSTER):
		kmeans = KMeans(n_clusters=2, random_state=0)

	if(doSVM):
		clf_linear = svm.SVC(kernel='linear')
		clf_poly = svm.SVC(kernel='poly')
		clf_radial = svm.SVC(kernel='rbf')

		clf_linear.fit(X, y)
		# clf_poly.fit(X, y)
		# clf_radial.fit(X, y)

		linear_train_accuracy = clf_linear.score(X, y)
		linear_test_accuracy = clf_linear.score(X_test, y_test)
		# poly_test_accuracy = clf_poly.score(X_test, y_test)
		# radial_test_accuracy = clf_radial.score(X_test, y_test)

		print("Linear SVM Training Accuracy {}".format(linear_train_accuracy))
		print("Linear SVM Test Accuracy {}".format(linear_test_accuracy))
		# print("Polynomial Accuracy {}".format(poly_test_accuracy))
		# print("Radial Accuracy {}".format(radial_test_accuracy))

	if(doRandomForest):
		clf = RandomForestClassifier(max_depth=2, random_state=0)
		clf.fit(X, y)
		print("Random Forest Feature Importance:")
		features = X_df.columns
		importances = clf.feature_importances_
		importances, features = zip(*sorted(zip(importances, features), reverse=True))
		print(pd.DataFrame([features, importances]))

		train_score = clf.score(X, y)
		test_score = clf.score(X_test, y_test)
		print("Random Forest Training Accuracy {}".format(train_score))
		print("Random Forest Test Accuracy {}".format(test_score))

	if(doDecisionTree):
		clf = tree.DecisionTreeClassifier(min_samples_split=200)
		clf = clf.fit(X, y)

		train_score = clf.score(X, y)
		test_score = clf.score(X_test, y_test)
		print("Decision Tree Training Accuracy {}".format(train_score))
		print("Decision Tree Test Accuracy {}".format(test_score))

	if(doNerualNetwork):
		sess = tf.InteractiveSession()
		# init  = tf.global_variables_initializer().run()


		# detailed output
		tf.logging.set_verbosity(tf.logging.INFO)

		config = {
			'learning_rate': 0.001,
			'epochs': 10000,
			'hidden_units': [11, 22, 11],
			'dropout': 0.2,
		}

		config_grid = {
			'learning_rate': [0.0001, 0.001, 0.01, 0.1],
			'epochs': [5000, 10000, 25000],
			'hidden_units': [[13, 26, 13], ],
			'dropout': [0.1, 0.5, 0.1, 0.2],
        },

		evaluation, nn_estimator = run_dnn(X,X_test,y,y_test, TARGET_DRUG, config)

		# predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		# 	x = {"x": np.array(y_test.data)}, 
		# 	num_epochs = 1, 
		# 	shuffle = False)
		# predictions = list(nn_estimator.predict(input_fn=predict_input_fn))

		# print(predictions)


		# dnn_sk_parameter_search(X,X_test,y,y_test, TARGET_DRUG, config_grid)

		# predict = nn_estimator.predict(x=X_test.as_matrix())
		# labels = tf.Variable(y_test.values, dtype=tf.float32)
		# # acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, 0), predictions=predict)
		# # print(sess.run([acc, acc_op]))
		# # print(sess.run([acc]))

