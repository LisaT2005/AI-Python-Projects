import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# importing dataset
fashion_mnist = keras.datasets.fashion_mnist
# splitting into train/test data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(train_images.shape)
# 60000 images, 28x28 pixels each
# print(train_images[0, 23, 23])
# grayscale image, 0 is black and 255 is white
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images/255.0
test_images = test_images/255.0
# scale grayscale numbers to be between 0 and 1

# sequential is the basic form of neural network
# passing from left to right
model = keras.Sequential([
    # Flatten lets us take 28x28 grid and make it a line of 728px
    # input layer
    keras.layers.Flatten(input_shape=[28, 28]),
    # hidden layer
    keras.layers.Dense(128, activation='relu'),
    # output layer (10 neurons bc 10 classes we want to output)
    # softmax makes all values add up to 1
    keras.layers.Dense(10, activation='softmax')

])

# optimizer is for gradient descent
# loss calculation function
# metrics is output we want from the network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10)
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
# print("Test accuracy =", test_acc)
# overfitting is when a neural network does worse on
# test data than train data because it "memorized" the train data

model.fit(train_images, train_labels, epochs=7)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print("Test accuracy =", test_acc)

predictions = model.predict(test_images)
# np.argmax gets largest num in probability dist.
print(class_names[np.argmax(predictions[0])])
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()