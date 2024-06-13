import json
import os

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel


# Load the configuration file
with open('config.json') as config_file:
    config = json.load(config_file)


IMAGE_DIR = config.get("IMAGE_DIR", "images")

# Initialize the feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model.eval()  # Put the model in evaluation mode


def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding


index = faiss.IndexFlatL2(768)  # embedding shape is 768
file_names = []
images_to_embed = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".jpeg", ".png"))
]

for image_name in tqdm(images_to_embed):
    
    embedding = get_image_embedding(image_name)

    index.add(np.array(embedding).astype(np.float32))

    file_names.append(image_name)

faiss.write_index(index, "index.bin")

with open("index.json", "w") as f:
    json.dump(file_names, f)
