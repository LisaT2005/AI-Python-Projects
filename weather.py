# hidden markov models

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow import feature_column as fc
import tensorflow_probability as tfp

tfd = tfp.distributions # shortcut for later
initial_distribution = tfd.Categorical(probs=[0.8, 0.2]) # 80% chance of being cold on 1st day
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5], [0.8, 0.2]])
# cold day has 30% chance of being followed by hot day, 20% chance of being followed by a cold day
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])
# on cold day mean temp = 0 and SD = 5, on hot day mean = 15 and SD = 10
# loc is mean, scale is SD

model = tfd.HiddenMarkovModel(initial_distribution, transition_distribution, observation_distribution, num_steps=7)
# steps is how many days we want to predict for (how many times we "run" the model
# we want to predict the avg. temp. for the next 7 days

mean = model.mean()
# this value is actually a partially defined tensor so e need to evaluate it to get output

with tf.compat.v1.Session() as s:
    print(mean.numpy())
    # prints expected temperatures of each day
