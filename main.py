# linear regression
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow import feature_column as fc

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# .head() gets first 5 entries of data set, preview of some columns
# print(dftrain.head())

y_train = dftrain.pop('survived')
# stores values of 'survived' column in y_train
y_eval = dfeval.pop('survived')

# find specific row with .loc[index]
# print(dftrain.loc[0], dfeval.loc[0])
# find specific column with ["column_name"]
# print(dftrain["age"])
# gives statistics on data set
# print(dftrain.describe())

# dftrain.age.hist(bins=20)
# plt.show()
# dftrain.sex.value_counts().plot(kind='barh')
# plt.show()
# dftrain["class"].value_counts().plot(kind="barh")
# plt.show()
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# plt.show()

# we look at the data and hardcode this in
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# feature columns are the values we feed to the linear regression model to make predictions
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    # get all unique values of a categorical var. in the data set
    # create array with all possible values, and add to feature_columns
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# sometimes datasets are extremely large so we need to load data in "batches", give 32 entries at once
# epochs: how many times the model is going to see the same data
# one downside to this is overfitting where the AI just memorizes the specific dataset
# but can't predict anything outside the training data
# to solve this start w/ low amount of epochs and increment as model improves

# create an input function to determine how data is going to be broken into epochs and batches
# usually don't need to code from scratch


def create_input_function(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function(): # returned by create_input_function
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        # create tf.data.Dataset object which is what TF needs for input
        if shuffle:
            ds.shuffle(1000) # randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs)
        # split dataset to batches of 32, repeat process for number of epochs
        return ds # return 1 batch of the dataset
    return input_function # return input function

train_input_fn = create_input_function(dftrain, y_train)
eval_input_fn = create_input_function(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn) # train
result = linear_est.evaluate(eval_input_fn) # gets model metrics/stats by testing on testing data (eval data)

# print(result)
# print("accuracy:", result['accuracy'])
# result is a dict of stats about the variable, calling the 'accuracy' key to get the value
result = list(linear_est.predict(eval_input_fn))
# print(dfeval.loc[3], "survived:", y_eval.loc[3])
# print("predicted probability of survival:", result[3]['probabilities'][1])
# first element of result, probabilities, prob. of surviving
