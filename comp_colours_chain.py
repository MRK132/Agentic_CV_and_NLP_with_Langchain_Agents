import os
import ast
import json
from typing import List
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from torch import Tensor
import faiss
from transformers import (
    AutoTokenizer,
    AutoModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from utilities import string_to_embedding, cls_pooling, display_images_similarity, display_images_complementary

# Load the configuration file
with open('config.json') as config_file:
    config = json.load(config_file)

BASE_URL = config["BASE_URL"]
API_KEY = config["API_KEY"]
MODEL_NAME = config["MODEL_NAME"]
NUM_SIMILAR_ITEMS = config.get("NUM_SIMILAR_ITEMS", 24)
IMAGE_DIR = config.get("IMAGE_DIR", "images")


class ComplementaryColoursApplication:
    def __init__(self):
        pass

    def run(self, customer_image_path, customer_question):

        model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model_text = AutoModel.from_pretrained(model_ckpt)
        device = torch.device("cpu")

        # load the text caption embedding database for similarity search, along with filepath index, and string captions for when plotting
        embedding_index = faiss.read_index(
            "text_index.bin"
        )

        with open(
            "text_index.json", "r"
        ) as f:
            file_names_index = json.load(f)

        with open(
            "text_index_captions.json", "r"
        ) as f:
            captions_index = json.load(f)

        # Define LangChain Agent tools:
        class ImageCaptionTool(BaseTool):
            name = "Image captioner"
            description = (
                "Use this tool when given the path to an image that you would like to be described. "
                "It will create a simple caption describing the image."
                "It will return the item of clothing found in the caption as a string for the next tool."
            )

            def _run(self, img_path):
                image = Image.open(img_path).convert("RGB")

                model_name = "Salesforce/blip-image-captioning-large"
                device = "cpu"

                processor = BlipProcessor.from_pretrained(model_name)
                model = BlipForConditionalGeneration.from_pretrained(model_name).to(
                    device
                )

                inputs = processor(image, return_tensors="pt").to(device)
                output = model.generate(**inputs, max_new_tokens=30)

                caption = processor.decode(output[0], skip_special_tokens=True)

                return caption

            def _arun(self, query: str):
                raise NotImplementedError("This tool does not support async")

        class ExtractColours(BaseTool):
            name = "Extract colours"
            description = (
                "Use this tool when asked to extract colours from text."
                "It will return the extracted colours at the end."
            )

            def _run(self, query_text):
                if type(query_text) == dict:
                    query_text = ", ".join(str(value)
                                           for value in query_text.values())

                llm = ChatOpenAI(
                    openai_api_base=BASE_URL,
                    model_name=MODEL_NAME,
                    openai_api_key=API_KEY,
                )

                message = llm.predict_messages(
                    [
                        HumanMessage(
                            content='In this customer question, what colour do they want to search for? Also give a list of complementary colours for that colour. Return the response as a dictionary, like so: {"colour": "", "complementary_colours": "" } Customer Question: <'
                            + query_text
                            + "> "
                        )
                    ]
                )
                print(message.content)
                str_dict = message.content
                actual_dict = ast.literal_eval(str_dict)

                print(type(actual_dict))
                print(actual_dict)
                return actual_dict

            def _arun(self, query: str):
                raise NotImplementedError("This tool does not support async")

        class SimilaritySearchTool(BaseTool):
            name = "Similarity searcher tool"
            description = (
                "Use this tool when asked to perform a similarity search for colours."
                "It will return the list of similar item ids at the end."
            )

            def _run(self, clothing_item, complementary_colours):
                model_text.to(device)

                if type(clothing_item) == dict:
                    clothing_item = ", ".join(
                        str(value) for value in clothing_item.values()
                    )

                print("converting string question to embedding:")

                query_text = f"{complementary_colours} colours?"
                print(query_text)
                embedding = (
                    string_to_embedding(
                        [query_text], tokenizer, model_text, device)
                    .cpu()
                    .detach()
                    .numpy()
                )

                print("reading from text embedding index")

                print("searching for similar text vectors")
                (
                    nearest_neighbor_distances,
                    nearest_neighbor_indices,
                ) = embedding_index.search(embedding, NUM_SIMILAR_ITEMS)

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

                most_similar_captions = [
                    captions_index[i] for i in nearest_neighbor_indices[0]
                ]

                filtered_images = []
                filtered_image_ids = []
                filtered_captions = []
                for image, image_id, caption in zip(
                    most_similar_images, most_similar_images_ids, most_similar_captions
                ):
                    if clothing_item not in caption:
                        print(f"{clothing_item} not in {caption}:")
                        filtered_images.append(image)
                        filtered_image_ids.append(image_id)
                        filtered_captions.append(caption)
                    else:
                        print(f"Filtered out {caption}")
                        continue

                display_images_complementary(
                    complementary_colours,
                    filtered_images,
                    filtered_image_ids,
                    filtered_captions,
                )
                return most_similar_images_ids

            def _arun(self, query: str):
                raise NotImplementedError("This tool does not support async")

        class CSVDatabaseChecker(BaseTool):
            name = "CSV database check tool"
            description = (
                "Use this tool when asked to check through a csv for items of a similar colour."
                "It will return the list of similar item ids at the end."
            )

            def _run(self, clothing_item: str, complementary_colours: list):
                if type(clothing_item) == dict:
                    clothing_item = ", ".join(
                        str(value) for value in clothing_item.values()
                    )

                if type(complementary_colours) == dict:
                    complementary_colours = complementary_colours[
                        "title"
                    ]  # = ', '.join(str(value) for value in complementary_colours.values())

                if type(complementary_colours) == str:
                    complementary_colours = complementary_colours.split(", ")

                df = pd.read_csv("articles.csv")
                # keep rows where 'product_description' does not contain mention of the input item
                df = df[
                    ~df["product_type_name"].str.contains(
                        clothing_item, case=False, na=False
                    )
                ]

                df = df[
                    ~df["garment_group_name"].str.contains(
                        clothing_item, case=False, na=False
                    )
                ]

                model_text.to(device)

                # Convert the entire 'colour_group_name' column to embeddings in a batch
                embeddings = (
                    string_to_embedding(
                        df["colour_group_name"].tolist(
                        ), tokenizer, model_text, device
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

                df["colour_embedding"] = list(embeddings)

                target_colours = complementary_colours
                print("target colours:")
                print(target_colours)

                # Encode the target colours to get the embeddings
                target_embeddings = (
                    string_to_embedding(
                        target_colours, tokenizer, model_text, device)
                    .cpu()
                    .detach()
                    .numpy()
                )

                # Function to compute dot product similarity
                def dot_product_similarity(embedding1, embedding2):
                    return np.dot(embedding1, embedding2)

                # Add a similarity score column for each target colour
                for colour, target_embedding in zip(target_colours, target_embeddings):
                    df[colour + "_similarity"] = df["colour_embedding"].apply(
                        lambda x: dot_product_similarity(x, target_embedding)
                    )

                # Number of top similar items to retrieve
                top_n = 24

                # Retrieve the top article IDs for each target colour
                top_article_ids = {}
                for colour in target_colours:
                    sorted_df = df.sort_values(
                        by=colour + "_similarity", ascending=False
                    )
                    top_article_ids[colour] = sorted_df.head(top_n)[
                        "article_id"
                    ].tolist()

                # Print the top article IDs for each colour
                for colour, ids in top_article_ids.items():
                    print(f"Top {top_n} article IDs for {colour}: {ids}")

                return top_article_ids

            def _arun(self, query: str):
                raise NotImplementedError("This tool does not support async")

        # initialize the agent
        tools = [
            ExtractColours(),
            ImageCaptionTool(),
            SimilaritySearchTool(),
            CSVDatabaseChecker(),
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
            # agent="chat-conversational-react-description",
            # Works better here, really needed multi-arg tools
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            max_iterations=5,
            verbose=True,
            memory=conversational_memory,
            early_stopping_method="generate",
        )

        user_question = f"Generate a caption for this image, and return the name of the clothing item present in the caption as a string. Next, extract the colours in this users question: {customer_question}. Next, perform a similarity search for the extracted colours. Finally, check through the CSV database for items of similar colours"
        response = agent.run(
            f"{user_question}, this is the image path: {customer_image_path}"
        )

        return response


def main():
    application = ComplementaryColoursApplication()
    customer_image_path = "/path/to/image.jpg"
    customer_question = ""
    response = application.run(customer_image_path, customer_question)
    return response


if __name__ == "__main__":
    main()
