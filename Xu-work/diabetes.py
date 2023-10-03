import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

pima_df = pd.read_csv("diabetes.csv")

pima_df.head()

pima_df.describe()

pima_df.info()

shape = pima_df.shape
print(shape)

pima_df.isnull().sum()

X = pima_df.drop("Outcome", axis=1)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

y = pima_df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter = 25000)

clf.fit(X_train, np.ravel(y_train))

y_pred = clf.predict(X_test)

y_true = y_test

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision Score:", precision)
print("Recall Score: ", recall)
print("F1 Score: ", f1_score)
