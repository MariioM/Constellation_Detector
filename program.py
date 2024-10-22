import pandas as pd
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt  # Importa matplotlib para graficar

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
            image = cv2.imread(image_path)
            image = cv2.resize(image, (SIZE_IMG, SIZE_IMG))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.reshape(SIZE_IMG, SIZE_IMG, 1)
            images.append(image)
            labels.append(label)
            training_data.append([image, label])
        else:
            print(f'Image not found: {image_path}')

    return images, labels

images, labels = load_data(dir_dataset, file_data)

# Normalize data
images = np.array(images).astype(float) / 255

# Convert labels to numerical values
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels, num_classes=88)  # Assuming 88 constellations

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

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
datagen.fit(X_train)

# Model
constellation_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE_IMG, SIZE_IMG, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(88, activation='softmax') 
])


# Compile model
constellation_model.compile(optimizer='adam', 
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

# Train model and store history
history = constellation_model.fit(datagen.flow(X_train, y_train, batch_size=32),
                                    validation_data=(X_val, y_val),
                                    epochs=50)

#Visualizar
plt.xlabel('# Epoca')
plt.ylabel("Magnitud de pérdida")
plt.plot(history.history["loss"])
plt.savefig('perdida_entrenamiento_numeros.png')

constellation_model.save('constellation_model.h5')