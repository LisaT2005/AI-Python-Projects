import tensorflow as tf
from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# manually split data into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)

# function to get name of label
get_label_name = metadata.features['label'].int2str

# display 2 images from the dataset
# for image, label in raw_train.take(2):
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(label))
#     plt.show()

# images are different dimensions so we need to scale them to be the same size
IMG_SIZE = 160
# usually better to make images smaller than bigger

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    #plt.show()

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# using model MobileNet V2 by Google as base model
# use predetermined weights form imagenet (Google dataset)
# include top is false because we only want 2 classifications (dog and cat)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# print(base_model.summary())

# freezing: disabling the training property of a layer
# so that we don't change the weights of the pretrained layers
base_model.trainable = False
# taking the average of all 1280 5x5 feature maps
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# only need 1 neuron to predict cats or dogs
prediction_layer = tf.keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# learning rate: how much you can modify the weights/biases of the network
# using BinaryCrossentropy because we only have 2 classes
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# evaluate model before training
initial_epochs = 3
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
print(loss0, accuracy0)

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

print(history.history['accuracy'])
# save model
model.save('dogs_vs_cats.h5')
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

