import os
import ast
import json
from langchain.schema import HumanMessage

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from restructure_image_files import move_images_to_main_directory

from similarity_search_chain import SimilaritySearchApplication
from comp_colours_chain import ComplementaryColoursApplication

# Restructure the dataset images if needed
move_images_to_main_directory("images")

def run_main_application():

    # Load the configuration file
    with open('config.json') as config_file:
        config = json.load(config_file)

    BASE_URL = config["BASE_URL"]
    API_KEY = config["API_KEY"]
    MODEL_NAME = config["MODEL_NAME"]
    USE_CASE_1_IMAGE_PATH = config["USE_CASE_1_IMAGE_PATH"]
    USE_CASE_2_IMAGE_PATH = config["USE_CASE_2_IMAGE_PATH"]

    #  User can specify this when running the application
    USER_QUESTION = os.environ.get("USER_QUESTION", "Use_Case_1")

    llm = ChatOpenAI(
        openai_api_base=BASE_URL,
        model=MODEL_NAME,
        openai_api_key=API_KEY,
    )

    if USER_QUESTION == "Use_Case_1":
        user_question = "I saw this picture on Instagram and I want the trousers she is wearing. What would you suggest?"
    elif USER_QUESTION == "Use_Case_2":
        user_question = "What items would go well with this product? I want autumn colours."
    else:
        raise ValueError(
            'The user question must be either "Use_Case_1" or "Use_Case_2". Please edit the config file and try again.')

    message = llm.predict_messages(
        [HumanMessage(
            content='In this user question, what is their desired application? Choose from either "similarity search" or "complementary colour items". If similarity search, return the item they want to search for, as well as a list of synonyms of that item. Return the response as a dictionary, like so: {"application": "", "item": "", synonyms_list": ["", ""] }. If complementary colour items, return the user question. Return the response as a dictionary, like so: {"application": "", "user_question": "" } Customer Question: <'+user_question+'> ')]
    )

    str_dict = message.content
    actual_dict = ast.literal_eval(str_dict)
    print(actual_dict)

    if actual_dict['application'] == "similarity search":

        item_with_synonyms_list = [
            actual_dict['item']] + actual_dict['synonyms_list']

        app = SimilaritySearchApplication()

        response = app.run(USE_CASE_1_IMAGE_PATH, item_with_synonyms_list)
        print(response)

    elif actual_dict['application'] == "complementary colour items":

        user_question = actual_dict["user_question"]

        app = ComplementaryColoursApplication()

        response = app.run(USE_CASE_2_IMAGE_PATH, user_question)
        print(response)

    else:
        raise ValueError(
            "I'm sorry, I'm not sure how to fulfil your request at this time.")
        # Hypothetically, if the user asked for something that would not have an application yet implemented,
        # we could display this.


if __name__ == "__main__":
    run_main_application()
