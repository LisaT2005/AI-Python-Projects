import tensorflow as tf
from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# convolutional neural network
# CIFAR image dataset
# 60000 images, 32x32, 10 classes, 6000 images of each class

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# pixel values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck']

# IMG_INDEX = 1
# plt.imshow(train_images[IMG_INDEX])
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()

model = models.Sequential()
# Convolutional layer (num. filters, sample size, activation function, input shape
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# MaxPooling2D(sample size, stride)
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# turn 4x4 map into 1D list
model.add(layers.Flatten())
# using dense layers to classify
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=4, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)