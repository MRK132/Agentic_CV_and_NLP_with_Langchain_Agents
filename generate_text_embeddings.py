import json
import os

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModel

"""
File to generate a caption for each image, generate a text embedding for each caption, and store them in a faiss index.

Also saves the index to the image filenames (product ids) and an index to the actual string captions for each image, which are used during plotting.

"""

# Load the configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

IMAGE_DIR = config.get("IMAGE_DIR", "images")


# Load the caption model and processor
caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

# Load the textmodel and tokenizer
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

device = torch.device("cpu")
model.to(device)


def process_and_caption_image(query_image, processor, model):
    # Process the image and generate a caption
    image = Image.open(query_image)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    caption = caption.replace("a close up of", "")  # remove "a close up of" from caption if present - dont need
    print(caption)
    return caption

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def string_to_embedding(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


index = faiss.IndexFlatL2(768)
file_names = []
captions = []
images_to_caption = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.endswith((".jpg", ".jpeg", ".png"))
]

for image_name in tqdm(images_to_caption):
    
    string_caption = process_and_caption_image(image_name, caption_processor, caption_model)
    captions.append(string_caption)

    embedding = string_to_embedding([string_caption]).cpu().detach().numpy()

    index.add(np.array(embedding).astype(np.float32))

    file_names.append(image_name)

faiss.write_index(index, "text_index.bin")

with open("text_index.json", "w") as f:
    json.dump(file_names, f)

with open("text_index_captions.json", "w") as f:
    json.dump(captions, f)
