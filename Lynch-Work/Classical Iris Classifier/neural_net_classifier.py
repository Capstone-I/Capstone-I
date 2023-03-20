import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path


def main():
	dataset_path = os.path.normpath("./data/iris.data")
	iris_train = pd.read_csv(dataset_path, names=["Sepal-Length", "Sepal-Width", "Petal-Length", "Petal-Width", "Name"])
	iris_train = iris_train.sample(frac=1).reset_index(drop=True)
	features = iris_train.copy()
	labels = features.pop("Name")
	
	# Create Preprocessing models
	
	# Features:
	feature_input_set = {}
	for name, column in features.items():
		dtype = column.dtype
		if dtype == object:
			dtype = tf.string
		else:
			dtype = tf.float64
		feature_input_set[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
	
	concat_layer = tf.keras.layers.Concatenate()(feature_input_set.values())
	norm = tf.keras.layers.Normalization()
	norm.adapt(features[feature_input_set.keys()])
	normalization_result = norm(concat_layer)
	normalization_model = tf.keras.Model(feature_input_set, normalization_result)
	
	# Labels
	lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(labels))
	
	body = tf.keras.Sequential([
		tf.keras.layers.Dense(64),
		tf.keras.layers.Dense(4)
	])
	
	preprocessing = normalization_model(feature_input_set)
	result = body(preprocessing)
	
	model = tf.keras.Model(feature_input_set, result)
	
	model.compile(optimizer='adam',
				  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])
	
	features_dict = {name: np.array(value) for name, value in features.items()}
	
	model.fit(x=features_dict, y=lookup(labels), epochs=50)

	test_path = os.path.normpath("./data/iris_test_data.data")
	iris_test = pd.read_csv(test_path, names=["Sepal-Length", "Sepal-Width", "Petal-Length", "Petal-Width", "Name"])
	iris_test = iris_test.sample(frac=1).reset_index(drop=True)
	test_features = iris_test.copy()
	test_labels = test_features.pop("Name")
	test_features_dict = {name: np.array(value) for name, value in test_features.items()}
	model.evaluate(x=test_features_dict, y=lookup(test_labels))
	

if __name__ == "__main__":
	main()
