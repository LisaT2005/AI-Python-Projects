# classification
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

# input has sepal length, sepal width, petal length, petal width
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    'iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
test_path = tf.keras.utils.get_file(
    'iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')

# reading datasets into pandas dataframe
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# in the data set 0 is Setosa, 1 is Versicolor, 2 is Virginica
train_y = train.pop('Species')
test_y = test.pop('Species')

print(train.head(), "\nshape", train.shape)


def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

my_feature_columns = []
# adding different possible values of var into feature column
# (unlike the titanic dataset we only have quantitative column so no need for 2 loops
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# tensorflow has DNNClassifier (Deep Neural Network), LinearClassifier, DNN is recommended by TF

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # two hidden layers of 30 and 10 nodes respectively
    hidden_units=[30, 10],
    # 3 possible values
    n_classes=3)

classifier.train(
    # lambda is an anonymous function that can be defined in 1 line, can be used to pass a function as an argument
    input_fn=lambda: input_fn(train, train_y, training=True),
    # go thru the data set until we have hit 5000 numbers
    steps=5000)

result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('Test set accuracy: {accuracy:0.3f}'.format(**result))

# writing code where the user can input sepal length, width, petal length, width, get output
def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Type in values:")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False
    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))

# predictions come back as dictionaries
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%)'.format(
    SPECIES[class_id], 100 * probability))

