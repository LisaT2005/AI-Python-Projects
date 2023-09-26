from keras.preprocessing import sequence
from keras.utils import pad_sequences
import keras
import tensorflow as tf
import os
import numpy as np

# goal is to make a model that predicts the next character in a sequence using RNN
# we only need 1 piece of data for training, shakespeare play extract
path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# import your own text data
# from google.colab import files
# path_to_file = list(files.upload().keys())[0]

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# print(text[:250])

# encoding each unique character as a different integer

vocab = sorted(set(text))

# turning vocab into dictionary and array
char2index = {u:i for i, u in enumerate(vocab)}
index2char = np.array(vocab)


def text_to_int(text):
    # loops through every character in txt, gets index and makes that into a list
    return np.array([char2index[c] for c in text])


text_as_int = text_to_int(text)
# print(text[:250])
# print(text_as_int[:250])


def int_to_text(ints):
    try:
        # converts into numpy array if it isn't already
        ints = ints.numpy()
    except:
        pass
    return ''.join(index2char[ints])


# print(int_to_text(text_as_int[:13]))

# need to split test data into shorter sequences to feed to the RNN
# example Hello divide into Hell and ello

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# drop_remainder means removing extra characters from a sequence if the length is bigger than 101
sequences = char_dataset.batch(batch_size=seq_length+1, drop_remainder=True)


def split_input_target(chunk): # example = 'hello'
    input_text = chunk[:-1] # 'hell'
    target_text = chunk[1:] # 'ello'
    return input_text, target_text


# .map() applies function to every target
dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# buffer size is used to shuffle the dataset
# TF doesn't shuffle in memory, instead uses a buffer
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        # we don't know how long the input is going to be which is why batch_input shape has None
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             # return intermediate stage at every step, more "verbose" output
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
# print(model.summary())

# model's output shape is an array of 64 arrays (shape is (64, 100, 165))
# for (batch size, sequence length, vocab size)
# there is a prediction for each time step which is why 1 training example has 100 outputs
# the numbers are the probability of each character occurring next
# we need to write our own loss function because there is no built-in analysis
# right now we have random weights and biases so it's pretty random


def loss(labels, logits):
    # logits = probability dist.
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)

# set up model to save checkpoints as it trains, we can load a model from a checkpoint and continue training it
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# history = model.fit(data, epochs=2, callbacks=[checkpoint_callback])
# model.save('RNN_play_generator.h5')

model = tf.keras.models.load_model('RNN_play_generator.h5', compile=False)

# we need to rebuild the model with batch size of 1 so that we can give it 1 input
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
# adding the checkpoint weights
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
