import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dir_dataset = 'dataset/'
file_data = 'Mapper.xlsx'

SIZE_IMG = 200

training_data = []

def load_data(dataset_dir, file):
    images = []
    labels = []
    data_file_path = os.path.join(dataset_dir, file)
    df = pd.read_excel(data_file_path)

    for i, row in df.iterrows():
        image_name = row["Image"].strip()
        label = row["Constellation"].strip()
        image_path = os.path.join(dataset_dir, 'train', image_name) 
        if os.path.exists(image_path):
            print(image_path)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (SIZE_IMG, SIZE_IMG))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.reshape(SIZE_IMG, SIZE_IMG, 1)
            images.append(image)
            labels.append(label)
            training_data.append([image, label])
        else:
            print(f'Image doesn´t found: {image_path}')

    return images, labels

images, labels = load_data(dir_dataset, file_data)

#Normalize data
images = np.array(images).astype(float) / 255
labels = np.array(labels)

# Data augmentation

datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
datagen.fit(images)

#Model (ToDo: Change model architecture)
constellation_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(80,80,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])