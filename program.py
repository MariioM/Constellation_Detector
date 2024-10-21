import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

dir_dataset = 'dataset/'
file_data = 'Mapper.xlsx'

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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            labels.append(label)
        else:
            print(f'Imagen no encontrada: {image_path}')

    return images, labels

images, labels = load_data(dir_dataset, file_data)

print(f'Cantidad de imágenes cargadas: {len(images)}')
print(f'Cantidad de etiquetas cargadas: {len(labels)}')

num_images = min(25, len(images))

fig, axes = plt.subplots(5, 5, figsize=(15, 15))
axes = axes.flatten()

for i in range(num_images):
    axes[i].imshow(images[i])
    axes[i].set_title(labels[i])
    axes[i].axis('off')

for j in range(num_images, 25):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
