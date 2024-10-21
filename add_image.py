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
        match = re.match(r'image(\d+)', image)
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
        if not re.match(r'image\d+', image_name):
            new_image_name = f'image{next_image_number}.jpg'
            print(f'Original image name: {image_name}, new image name: {new_image_name}')
            
            print('Select a constellation:')
            for idx, constellation in enumerate(constellations):
                print(f'{idx + 1}. {constellation}')
            
            while True:
                try:
                    choice = int(input('Introduce constellation´s number: '))
                    if 1 <= choice <= len(constellations):
                        label = constellations[choice - 1]
                        break
                    else:
                        print('Not a valid option.')
                except ValueError:
                    print('Insert a valid number.')
            
            new_row = pd.DataFrame({"Image": [new_image_name], "Constellation": [label]})
            df = pd.concat([df, new_row], ignore_index=True)
            
            original_image_path = os.path.join(dir_dataset, image_name)
            new_image_path = os.path.join(dir_dataset, new_image_name)
            os.rename(original_image_path, new_image_path)

            next_image_number += 1

new_images = [f for f in os.listdir(dir_dataset) if f.endswith(('.png', '.jpg', '.jpeg'))]

if new_images:
    add_new_images(new_images)
else:
    print('There are not new images.')

df.to_excel(excel_path, index=False)
print('Excel file updated.')
