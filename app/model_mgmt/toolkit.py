# flake8: noqa --E501

import bigframes.pandas as bpd
import logging
import json
import os

from google.cloud import bigquery
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from json_repair import repair_json
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    grounding,
    Tool
)
from vertexai.language_models import TextEmbeddingModel


import model_mgmt.testing as testing
project_id = testing.project_id
location = testing.location


# project_id = os.environ.get("PROJECT_ID")
# location = os.environ.get("LOCATION")
# project_id = "scg-l200-genai2"
# location = "us-west1"


generation_config = GenerationConfig(
    temperature=0.1,
    top_p=.3,
    top_k=1,
    candidate_count=1,
    max_output_tokens=200,
)

model = GenerativeModel(
    model_name="gemini-1.5-flash-002",
    generation_config=generation_config,
    # safety_settings=config.safety_settings
)


# @tool
def parse_input(user_input: str):
    """
    Determine the User_Destination and Points_of_Interest the user has identified in the chat

    Args:
        user_input: A string of user inputs and chat responses

    Returns:
        A JSON object that describes the user's key requirements for the trip
    """
    prompt = f"""
    Your sole purpose is to identify key information in a user's request.
    A user is asking for travel information to a specific location and points of interest.
    Parse the user's input to identify the following:
        * User_Request_Destination (city or country they want to visit)
        * Points_of_Interest (specific places they want to see)
        * User_Interests (type of activities the user wants to focus on, like shopping or historical sites)

    User input: {user_input}

    Return the information in JSON format with the following structure:
    {{
        "User_Destination": "destination_city_or_country",
        "Points_of_Interest": ["point_of_interest_1", "point_of_interest_2"],
        "User_Interests": "user_interests"
    }}
    """

    response = model.generate_content(prompt)
    try:
        cleaned_response = repair_json(response.text)
    except ValueError as e:
        print(f"Invalid JSON output from the model: {response.text}")
        # Return a default empty result or handle the error appropriately
        cleaned_response = {"User_Destination": None, "Points_of_Interest": []}
    return cleaned_response

def get_text_embeddings(text_input):
    text_embed_model = TextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003")
    text_embeddings = text_embed_model.get_embeddings([text_input])
    for embedding in text_embeddings:
        vector = embedding.values
    return vector


# @tool
def hotel_search(pois):
    """
    Search the database of hotels to find a hotel near the specified Points_of_Interest in Florence, Italy

    Args:
        pois: A list of Points_of_Interest

    Returns:
        A JSON object with the name, description, and address of the recommended hotel
    """
    client = bigquery.Client()
    poi_vector = get_text_embeddings(pois)
    vector_list = ""
    for vector in vector_list:
        vector_list += vector + ", "
    bpd.options.bigquery.project = project_id
    bpd.options.bigquery.location = location
    vector_search_options = '\'{"use_brute_force":true}\''
    hotel_search_query = f'''
        CREATE TEMP TABLE hold AS
        SELECT
            array(select cast(elem as float64) from unnest(split("{vector_list}, ",") elem) AS poi_vector;

        with search AS (
            SELECT
                query.hotel_name AS hotel_name,
                query.hotel_address AS hotel_address,
                query.hotel_description AS hotel_description
            FROM
                VECTOR_SEARCH( TABLE hold,
                    'poi_vector',
                    TABLE `hotels.florence_embeddings`,
                    'nearest_attractions_embeddings',
                    top_k => 5,
                    distance_type => 'COSINE',
                    OPTIONS => {vector_search_options})
            ORDER BY
                distance
                )

    SELECT * FROM search LIMIT 1
    '''
    hotel = bpd.read_gbq(hotel_search_query).to_dict(orient="records")[0]
    hotel_json = json.dumps(hotel)
    logging.info(hotel_json)
    return hotel_json


parse_input_func = FunctionDeclaration(
    name="parse_input",
    description="Determine the destination and points of interest the user has identified",
    parameters={
        "type": "object",
        "properties": {
            "User_Destination": {"type": "string", "description": "Destination of the user wants to plan a trip for"},
            "Points_of_Interest": {"type": "array", "description": "List of points of interest the user wants to see"},
            "User_Interest": {"type": "string", "description": "Type of activities of the user wants to plan a trip for"},
        },
    },
)

hotel_search_func = FunctionDeclaration(
    name="hotel_search",
    description="Search the hotels database to find a hotel near the user's Points of Interest in Florence, Italy",
    parameters={
        "type": "object",
        "properties": {
            "User_Destination": {"type": "string", "description": "Destination in the format City, Country the user wants to plan a trip for"},
            "Points_of_Interest": {"type": "string", "description": "Comma-separated list of points of interest the user wants to see"},
        },
    },
)

tool_list = [parse_input_func, hotel_search_func]

tools = Tool(
    function_declarations=tool_list
)


# tool_node = ToolNode(tools)
