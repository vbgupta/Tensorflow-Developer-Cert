# Lung Image Multiclass Classification
# All these images are black and white

## 3 Classes
### 1. Normal, 2. Virus, 3. Bacteria

# Importing Libs
import pathlib
import cv2
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import PIL
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import warnings
warnings.filterwarnings("ignore")

class Lungclassification:

    def __init__(self, data, train_dir, test_dir):

        self.data = data
        self.train_dir = train_dir
        self.test_dir = test_dir

    def images(self):

        image_count = len(list(self.data.glob('*/*/*.jpeg')))
        print("Total Number of Images in the Test Folder:", image_count)
        return image_count

    def create_data(self):

        BATCH_SIZE = 32
        IMG_HEIGHT = 180
        IMG_WIDTH = 180

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            validation_split = 0.2,
            subset = "training",
            seed = 44,
            image_size = (IMG_HEIGHT, IMG_WIDTH),
            batch_size = BATCH_SIZE

        )


        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            validation_split = 0.2,
            subset = "validation",
            seed = 44,
            image_size = (IMG_HEIGHT, IMG_WIDTH),
            batch_size = BATCH_SIZE

        )
        CLASS_NAMES = train_ds.class_names
        print("Class Names:", CLASS_NAMES)
        return train_ds, val_ds, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES

    def preprocessing(train_ds, val_ds):

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # Notice the pixels values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image))
        return train_ds, val_ds

    def create_model(IMG_HEIGHT, IMG_WIDTH):

        NUM_CLASSES = 3

        model = Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(NUM_CLASSES)])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        print(model.summary())
        return model

    def train_model(model, train_ds, val_ds):

        EPOCHS = 10

        history = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs = EPOCHS

        )

        return history, EPOCHS

    def visualize_results(history, EPOCHS):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(EPOCHS)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def make_predictions(model, test_dir, CLASS_NAMES):

        for images in os.listdir(test_dir):

            images = os.path.join(test_dir + '//' + images)

            images = keras.preprocessing.image.load_img(
                images,
                target_size = (180, 180))

            img_array = keras.preprocessing.image.img_to_array(images)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                    .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))
            )

data = pathlib.Path(r"C:\Users\vbgup\CODE\PROJECTS\LungScans")
train_dir = pathlib.Path(r"C:\Users\vbgup\CODE\PROJECTS\LungScans\train\train")
test_dir = "C:/Users/vbgup/CODE/PROJECTS/LungScans/test/test"
classify = Lungclassification(data, train_dir, test_dir)
images = Lungclassification.images(classify)
train_ds, val_ds, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES = Lungclassification.create_data(classify)
train_ds, val_ds = Lungclassification.preprocessing(train_ds, val_ds)
sequential_model = Lungclassification.create_model(IMG_HEIGHT, IMG_WIDTH)
training, epochs = Lungclassification.train_model(sequential_model, train_ds, val_ds)
plot = Lungclassification.visualize_results(training, epochs)
preds = Lungclassification.make_predictions(sequential_model, test_dir, CLASS_NAMES)

