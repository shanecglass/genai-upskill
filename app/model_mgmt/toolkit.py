# flake8: noqa --E501

import testing
import os
import sys
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import bigframes.pandas as bpd
import logging
import json
import os

from google.cloud import bigquery
from langchain_core.prompts import ChatPromptTemplate
from json_repair import repair_json
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Tool
)
from vertexai.language_models import TextEmbeddingModel


project_id = testing.project_id
location = testing.location

# project_id = os.environ.get("PROJECT_ID")
# location = os.environ.get("LOCATION")


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
def get_user_preferences(user_email: str, project_id: str = project_id):
    """
    Identify the user's travel preferences

    Args:
        user_input: User's request

    Returns:
        A JSON object with user's first name, location, budget range, destination types, trip length, travel companions, and nearest airports
    """
    # prompt_template = """
    #     Extract the User_Email from the User_Request.

    #     User_Request: {user_input}

    #     ```json
    #     {{
    #         "User_Email": "...",
    #     }}
    #     ```
    # """
    # prompt = ChatPromptTemplate.from_template(prompt_template)
    # response = model.generate_content(
    #     prompt.format_messages(user_input=user_input))

    # try:
    #     cleaned_response = repair_json(response.text)
    # except ValueError as e:
    #     print(f"Invalid JSON output from the model: {response.text}")
    #     return "I don't recognize that email. Try entering it again"

    user_preferences_query = f'''
        SELECT
            first_name AS first_name,
            user_location AS location,
            user_budget_range AS budget_range,
            destination_types AS dest_types,
            trip_length AS trip_length,
            travel_companions AS travel_companions,
            STRING_AGG(DISTINCT(x.name), ", ") AS closest_airports
        FROM
            `{project_id}.travel_chat.users` users
        JOIN
            `{project_id}.travel_chat.user_airports` airports USING (user_id, user_name),
            UNNEST(user_airports) x
        WHERE
            user_email = "{user_email}"
        GROUP BY
            user_id,
            first_name,
            location,
            budget_range,
            dest_types,
            trip_length,
            travel_companions
    '''

    user_preferences = bpd.read_gbq(
        user_preferences_query).to_dict(orient="records")[0]
    user_preferences_json = json.dumps(user_preferences)
    logging.info(user_preferences_json)
    print(user_preferences_json)
    print(user_preferences_json)
    return user_preferences_json

def get_text_embeddings(text_input):
    text_embed_model = TextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003")
    # text_embeddings = text_embed_model.get_embeddings([text_input])
    text_embeddings = text_embed_model.get_embeddings([text_input])
    for embedding in text_embeddings:
        vector = embedding.values
    return vector


# @tool
# def hotel_search(pois):
#     """
#     Find a hotel near the user's Points_of_Interest in Florence, Italy

#     Args:
#         pois: A string containing a comma-separated list of Points_of_Interest

#     Returns:
#         A JSON object with the name, description, and address of the recommended hotel
#     """
#     client = bigquery.Client()
#     print("checking BQ")
#     poi_vector = get_text_embeddings(pois)
#     vector_str = ""
#     for vector in poi_vector:
#         vector_str += str(vector) + ", "
#     vector_str = vector_str.rstrip(', ').strip()
#     bpd.options.bigquery.project = project_id
#     bpd.options.bigquery.location = location
#     vector_search_options = '\'{"use_brute_force":true}\''
#     hotel_search_query = f'''
#         CREATE TEMP TABLE hold AS
#         SELECT
#             array(select cast(elem as float64) from unnest(split(TRIM("{vector_str}", ", "), ",")) elem) AS poi_vector;

#         with search AS (
#             SELECT
#                 query.hotel_name AS hotel_name,
#                 query.hotel_address AS hotel_address,
#                 query.hotel_description AS hotel_description
#             FROM
#                 VECTOR_SEARCH( TABLE hold,
#                     'poi_vector',
#                     TABLE `hotels.florence_embeddings`,
#                     'nearest_attractions_embeddings',
#                     top_k => 5,
#                     distance_type => 'COSINE',
#                     OPTIONS => {vector_search_options})
#             ORDER BY
#                 distance
#                 )

#     SELECT * FROM search LIMIT 1
#     '''
#     hotel = bpd.read_gbq(hotel_search_query).to_dict(orient="records")[0]
#     hotel_json = json.dumps(hotel)
#     logging.info(hotel_json)
#     print(hotel_json)
#     return hotel_json


get_user_preferences_func = FunctionDeclaration(
    name="get_user_preferences",
    description="""
    Identify the user's travel preferences based on the email identified in the user's input

    Args:
        user_input: User's input

    Returns:
        A JSON object with user's first name, location, budget range, destination types, trip length, travel companions, and nearest airports
    """,
    parameters={
        "type": "object",
        "properties": {
            "User_Email": {"type": "string", "description": "User's email address"},
        },
    },
)

# hotel_search_func = FunctionDeclaration(
#     name="hotel_search",
#     description="""
#         Find a hotel near the user's Points_of_Interest in Florence, Italy

#         Args:
#             pois: A string containing a comma-separated list of Points_of_Interest

#         Returns:
#             A JSON object with the name, description, and address of the recommended hotel
#     """,
#     parameters={
#         "type": "object",
#         "properties": {
#             "User_Destination": {"type": "string", "description": "Destination in the format City, Country the user wants to plan a trip for"},
#             "Points_of_Interest": {"type": "string", "description": "Comma-separated list of points of interest the user wants to see"},
#             # "User_Interests": {"type": "string", "description": "Type of activities of the user wants to plan a trip for"},
#         },
#         "required": ["User_Destination", "Points_of_Interest"],
#     },
# )

tool_list = [
    # hotel_search_func,
    get_user_preferences_func,
]

tools = Tool(
    function_declarations=tool_list
)


# tool_node = ToolNode(tools)
