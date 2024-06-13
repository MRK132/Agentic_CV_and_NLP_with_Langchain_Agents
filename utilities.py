import torch
from typing import List
from matplotlib import patches, pyplot as plt
from PIL import Image
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

# Define utility functions:


def cls_pooling(model_output: Tensor) -> Tensor:
    """
    Extracts the [CLS] token embedding as the pooled representation from the model's output.

    Args:
        model_output (Tensor): The output of the model which contains the hidden states.

    Returns:
        Tensor: The [CLS] token embedding.
    """
    return model_output.last_hidden_state[:, 0]


def string_to_embedding(
    text_list: List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
) -> Tensor:
    """
    Converts a list of strings into embeddings using the tokenizer and text embedding model.

    Args:
        text_list (List[str]): A list of strings to be converted into embeddings.

    Returns:
        Tensor: The embeddings corresponding to the input strings.

    """
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


def display_images_complementary(
    query_caption: str,
    image_paths: List[str],
    distances: List[float],
    subtitles: List[str],
) -> None:
    """
    Displays a grid of images that are complementary to a query image, with captions and distances as subtitles.

    Args:
        query_caption (str): The caption for the query image.
        image_paths (List[str]): A list of paths to the similar images.
        distances (List[float]): A list of distances or similarity scores for the similar images.
        subtitles (List[str]): A list of subtitles to be displayed under each image.
    """
    cols = 5  # Set the number of columns to 5
    rows = 5  # Set the number of rows to 5

    # Calculate total number of subplots needed (one extra for the query text)

    total_slots = rows * cols
    total_slots_with_query = total_slots + 1  # Including the slot for the query text

    # Create figure with subplots
    plt.figure(figsize=(cols * 3, rows * 3))
    plt.suptitle("Similar images to your query:")

    # Add the query text in the first position
    ax = plt.subplot(rows, cols, 1)
    plt.title(f"Query:")
    plt.text(
        0.5,
        0.5,
        query_caption,
        fontsize=12,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    plt.axis("off")

    # Start filling subplots from position 2
    subplot_idx = 2

    # Loop through all the images and plot them in the remaining slots
    for i, (image_path, distance, subtitle) in enumerate(
        zip(image_paths, distances, subtitles)
    ):
        # Check if we still have slots left to fill in the grid
        if subplot_idx <= total_slots_with_query:
            plt.subplot(rows, cols, subplot_idx)
            image = Image.open(image_path)
            plt.imshow(image)
            plt.title(f"{distance}\n Item ID: {subtitle}", fontsize=6)
            plt.axis("off")
            subplot_idx += 1

        # If we've filled all available slots, stop plotting more images
        if subplot_idx > total_slots_with_query:
            break

    plt.show()


def display_images_similarity(
    query_image_path: str,
    image_paths: List[str],
    distances: List[float],
    subtitles: List[str],
) -> None:
    """
    Displays the query image and a series of images that are similar to it, with distances and subtitles.

    Args:
        query_image_path (str): The file path to the query image.
        image_paths (List[str]): A list of file paths to the similar images.
        distances (List[float]): A list of distances or similarity scores for each of the similar images.
        subtitles (List[str]): A list of subtitles (typically product IDs) to be displayed under each image.
    """
    total_images = len(image_paths) + 1  # Including the query image
    cols = total_images  # Set the number of columns to the total number of images
    rows = 1  # All images will be in one row

    plt.figure(figsize=(cols * 5, rows * 5))
    plt.suptitle("Similar images to your query:")

    # Display the query image in the first position
    ax = plt.subplot(rows, cols, 1)
    query_image = Image.open(query_image_path)
    plt.imshow(query_image)
    plt.title("Query Image")

    plt.axis("off")

    # Create a red border around the query image using a patch
    red_border = patches.Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(red_border)

    # Display each similar image and distance
    for i, (cropped_in_image_path, distance, subtitle) in enumerate(
        zip(image_paths, distances, subtitles), start=1
    ):
        plt.subplot(rows, cols, i + 1)  # Increment subplot index
        image = Image.open(cropped_in_image_path)
        plt.xlabel(f"Product ID:\n {subtitle}")
        plt.imshow(image)
        plt.title(f"Product ID: \n {subtitle} \n \nN.N. Distance:\n {distance:.2f}")
        plt.axis("off")

    plt.tight_layout(w_pad=8)
    plt.show()
