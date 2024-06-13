import ast
import json
import os

import faiss
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection, ViTFeatureExtractor, ViTModel

from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool

from utilities import display_images_similarity


# Load the configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

BASE_URL = config["BASE_URL"]
API_KEY = config["API_KEY"]
MODEL_NAME = config["MODEL_NAME"]
NUM_SIMILAR_IMAGES = config.get("NUM_SIMILAR_IMAGES", 15)


class SimilaritySearchApplication:
    def __init__(self):
        pass

    def run(self, customer_img_path, item_with_synonms_list):

        class ObjectDetectionTool(BaseTool):
            name = "Object detector"
            description = (
                "Use this tool when given the path to an image that you would like to detect objects. "
                "It will return a list of all detected objects. Each element in the list in the format: "
                "[x1, y1, x2, y2] class_name confidence_score."
            )

            def _run(self, img_path):

                image = Image.open(img_path).convert("RGB")

                from transformers import (
                    AutoFeatureExtractor,
                    AutoModelForObjectDetection,
                )

                processor = AutoFeatureExtractor.from_pretrained(
                    "valentinafeve/yolos-fashionpedia"
                )
                model = AutoModelForObjectDetection.from_pretrained(
                    "valentinafeve/yolos-fashionpedia"
                )

                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)

                # convert outputs (bounding boxes and class logits) to COCO API
                target_sizes = torch.tensor([image.size[::-1]])
                results = processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.9
                )[0]

                detections = ""
                for score, label, box in zip(
                    results["scores"], results["labels"], results["boxes"]
                ):
                    detections += "[{}, {}, {}, {}]".format(
                        int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    )
                    detections += " {}".format(
                        model.config.id2label[int(label)])
                    detections += " {}\n".format(float(score))

                return detections

            def _arun(self, query: str):
                raise NotImplementedError("This tool does not support async")

        class CropImageAndSaveTool(BaseTool):
            name = "Crop and save image"
            description = (
                "Use this tool when are asked crop and save an image."
                "It will use the bounding box co-ordinates [x1, y1, x2, y2] to crop and save this image to disk."
                "This tool only accepts one input - detections."
            )

            def _run(
                self, detections
            ):  # Required new Agent, as needed more than one argument

                image = Image.open(customer_img_path).convert("RGB")

                # Coordinates for the object detection result
                # PIL expects a tuple (left, upper, right, lower)
                print(type(detections))

                if type(detections) == str:
                    detections = ast.literal_eval(detections)

                print(f"attempting to crop with {detections}")
                box = detections

                # Crop the image based on the coordinates
                cropped_image = image.crop(box)

                # Save the cropped image to disk
                cropped_image.save("cropped_image.jpg")

                print("Image cropped and saved to disk.")

                return "cropped_image.jpg"

            def _arun(self, query: str):
                raise NotImplementedError("This tool does not support async")

        class PerformSimilaritySearchTool(BaseTool):
            name = "Perform a similarity search for an image"
            description = (
                "Use this tool when asked to perform a similarity search for an image against a database of image embeddings."
                "It will return the list of similar image ids at the end."
            )

            def _run(self, cropped_image_path):
                # Initialize the feature extractor and model
                feature_extractor = ViTFeatureExtractor.from_pretrained(
                    "google/vit-base-patch16-224-in21k"
                )
                model = ViTModel.from_pretrained(
                    "google/vit-base-patch16-224-in21k")
                model.eval()  # Put the model in evaluation mode

                def get_image_embedding(cropped_image_path):
                    image = Image.open(cropped_image_path)

                    inputs = feature_extractor(
                        images=image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu(
                    ).numpy()
                    return embedding

                embedding_index = faiss.read_index(
                    "index.bin"
                )

                with open(
                    "index.json", "r"
                ) as f:
                    file_names_index = json.load(f)

                query = get_image_embedding(cropped_image_path)

                (
                    nearest_neighbor_distances,
                    nearest_neighbor_indices,
                ) = embedding_index.search(
                    np.array(query).astype(np.float32), NUM_SIMILAR_IMAGES
                )

                most_similar_images = [
                    file_names_index[i]
                    for i in nearest_neighbor_indices[0]
                ]
                print("most similar images:")
                print(most_similar_images)

                most_similar_images_ids = [
                    os.path.splitext(os.path.basename(image))[0]
                    for image in most_similar_images
                ]
                print("most similar image product ids:")
                print(most_similar_images_ids)

                verified_most_similar_images = []
                verified_nearest_neighbor_distances = []
                verified_most_similar_images_ids = []

                processor = AutoFeatureExtractor.from_pretrained(
                    "valentinafeve/yolos-fashionpedia"
                )
                model = AutoModelForObjectDetection.from_pretrained(
                    "valentinafeve/yolos-fashionpedia"
                )
                for im_path, nn_distance, image_id in zip(
                    most_similar_images,
                    nearest_neighbor_distances[0],
                    most_similar_images_ids,
                ):
                    im = Image.open(im_path)

                    inputs = processor(images=im, return_tensors="pt")
                    outputs = model(**inputs)

                    target_sizes = torch.tensor([im.size[::-1]])
                    results = processor.post_process_object_detection(
                        outputs, target_sizes=target_sizes, threshold=0.9
                    )[0]
                    print(results["labels"])
                    label_detections = ""
                    label_detections = [
                        model.config.id2label[int(label)] for label in results["labels"]
                    ]
                    print("label_detections")
                    print(label_detections)
                    print("item_with_synonms_list")

                    print(item_with_synonms_list)

                    if any(
                        word.lower() in label_detections
                        for word in item_with_synonms_list
                    ):
                        print(
                            f"Verified {im_path} for one of {item_with_synonms_list}")
                        verified_most_similar_images.append(im_path)
                        verified_nearest_neighbor_distances.append(nn_distance)
                        verified_most_similar_images_ids.append(image_id)
                    else:
                        print(
                            f"Skipping {im_path} because it is not verified.")
                        continue

                display_images_similarity(
                    cropped_image_path,
                    verified_most_similar_images,
                    verified_nearest_neighbor_distances,
                    verified_most_similar_images_ids,
                )

                return most_similar_images_ids

            def _arun(self, query: str):
                raise NotImplementedError("This tool does not support async")

        # initialize the agent, define available tools:
        tools = [
            ObjectDetectionTool(),
            CropImageAndSaveTool(),
            PerformSimilaritySearchTool(),
        ]

        conversational_memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=5, return_messages=True
        )

        llm = ChatOpenAI(
            openai_api_base=BASE_URL,
            model_name=MODEL_NAME,
            openai_api_key=API_KEY,
            
        )

        agent = initialize_agent(
            # Performs better at this task, more reliable over all
            agent="chat-conversational-react-description",
            # agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            max_iterations=5,
            verbose=True,
            memory=conversational_memory,
            early_stopping_method="generate",
        )

        user_question = f"Please tell me the bounding box of the detected trousers object in the image. This is the image path: {customer_img_path}. I would then like this image to be cropped and saved using the bounding box. Then, please run a similarity search for this cropped image."
        detections = agent.run(f"{user_question} .")
        print(detections)


def main():
    application = SimilaritySearchApplication()
    customer_img_path = "/path/to/image.jpg"
    item_with_synonms_list = ""
    response = application.run(customer_img_path, item_with_synonms_list)
    print(response)


if __name__ == "__main__":
    main()
