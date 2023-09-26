# recurring neural network (sentiment analysis)

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import pad_sequences
import tensorflow as tf

import os
import numpy as np

# movie review dataset has 25000 reviews preprocessed/labeled
# encoding system where integer of a word represents how common the word is in the dataset

# num. different words
VOCAB_SIZE = 88584
# max num. words in a review
MAXLEN = 250
BATCH_SIZE = 64
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words= VOCAB_SIZE)

# shows a list of numbers (the encoded words)
# print(train_data[0])

# if number of words is greater than 250 trim off extra words, if less than add 0s to make it 250
# treating test data as sequence which is why we pad sequences
train_data = pad_sequences(train_data, MAXLEN)
test_data = pad_sequences(train_data, MAXLEN)

# 0s are now added to the beginning so the length is 250
# print (train_data[0])
model = tf.keras.Sequential([
    # embedding layer (words turn into 32D vectors)
    # although data is already preprocessed embedding layer helps with grouping similar words together
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    # LSTM layer (recurrent)
    tf.keras.layers.LSTM(32),
    # want values 0 or 1 (positive or negative), sigmoid outputs between 0 and 1
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# print(model.summary())

model.compile(
    loss="binary_crossentropy",
    optimizer="rmsprop",
    metrics=['acc']
)

# validation split uses 20% of the training data to evaluate the models
# history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# results = model.evaluate(test_data, test_labels)
# print(results)
# model.save('sentiment_analysis_imdb.h5')

new_model = tf.keras.models.load_model('sentiment_analysis_imdb.h5')
# results = new_model.evaluate(test_data, test_labels)
# print(results)
word_index = imdb.get_word_index()


# function to encode text so we can make predictions
def encode_text(text):
    # converts sentence into list of words
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    # loops through tokens, if word is in imdb word index assign that index, otherwise assign 0
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    # pad token sequence and return 1st list (pad_sequences makes a 2d array)
    return pad_sequences([tokens], MAXLEN)[0]


text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
# print(encoded)

reverse_word_index = {value: key for (key, value) in word_index.items()}
# loops through word_index dictionary and makes a new dictionary but with the words reversed


def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "

    return text[:-1]
    # everything except last space


def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = new_model.predict(pred)
    print(result[0])

pos_review = "That movie was so awesome! I loved it and would watch it again because it was amazingly great"
print(predict(pos_review))
neg_review = "That movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
print(predict(neg_review))

