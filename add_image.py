import pandas as pd
import os
import json
import re

dir_dataset = 'dataset/train/'
file_data = 'Mapper.xlsx'
json_file = 'constellations.json'
excel_path = os.path.join('dataset', file_data)

if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
else:
    df = pd.DataFrame(columns=["Image", "Constellation"])

with open(json_file, 'r') as f:
    constellations_data = json.load(f)

constellations = [constellation['name'] for constellation in constellations_data]

def get_next_image_number(existing_images):
    max_number = 0
    for image in existing_images:
        match = re.search(r'image(\d+)', image)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
    return max_number + 1

def add_new_images(new_images):
    global df
    
    existing_images = df["Image"].tolist()
    next_image_number = get_next_image_number(existing_images)

    for image_name in new_images:
        new_image_name = f'image{next_image_number}.jpg'
        print(f'Imagen original: {image_name}, nueva imagen: {new_image_name}')
        
        print('Selecciona una constelación:')
        for idx, constellation in enumerate(constellations):
            print(f'{idx + 1}. {constellation}')
        
        while True:
            try:
                choice = int(input('Introduce el número de la constelación: '))
                if 1 <= choice <= len(constellations):
                    label = constellations[choice - 1]
                    break
                else:
                    print('Opción no válida. Intenta de nuevo.')
            except ValueError:
                print('Por favor, introduce un número válido.')
        
        df = df.append({"Image": new_image_name, "Constellation": label}, ignore_index=True)
        
        original_image_path = os.path.join(dir_dataset, image_name)
        new_image_path = os.path.join(dir_dataset, new_image_name)
        os.rename(original_image_path, new_image_path)

        next_image_number += 1

new_images = [f for f in os.listdir(dir_dataset) if f.endswith(('.png', '.jpg', '.jpeg'))]

if new_images:
    add_new_images(new_images)
else:
    print('No hay nuevas imágenes para agregar.')

df.to_excel(excel_path, index=False)
print('Archivo Excel actualizado.')
