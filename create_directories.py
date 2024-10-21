import os
import json

with open('./constellations.json', 'r') as file:
    data = json.load(file)

names = [element['name'] for element in data]

for name in names:
    os.mkdir('datasets/'+name)

