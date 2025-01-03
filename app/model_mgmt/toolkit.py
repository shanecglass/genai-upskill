# flake8: noqa --E501



import bigframes

import bigframes.pandas as bpd
import googlemaps
import logging
import json
import pandas as pd
import requests
# from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode


from google.cloud import bigquery, secretmanager
from langchain.tools import tool
from json_repair import repair_json
from retry_requests import retry
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Tool
)
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel


import testing as testing
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


def get_text_embeddings(text_input: str):
    text_embed_model = TextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003")
    # text_embeddings = text_embed_model.get_embeddings([text_input])
    text_embeddings = text_embed_model.get_embeddings(text_input)
    for embedding in text_embeddings:
        vector = embedding.values
    return vector


def get_multimodal_embeddings(input: str, dimension: int | None = 1408, img: str = None) -> list[float]:
    mm_embed_model = MultiModalEmbeddingModel.from_pretrained(
        "multimodalembedding@001")
    text_input = " ".join(input)
    if img is None:
        embedding = mm_embed_model.get_embeddings(
            contextual_text=text_input,
            dimension=dimension,
        )
    return embedding.text_embedding


def get_gmaps_key(secret_id: str, version_id: str = "latest"):
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version.
    response = client.access_secret_version(name=name)

    # Return the decoded payload.
    return response.payload.data.decode('UTF-8')


@tool
def parse_input(user_input: str) -> json:
    """
    Determine the user_destination and points_of_interest

    Args:
        user_input (str): User's request

    Returns:
        json: A JSON object with user_destination, points_of_interest, and user_interests
    """
    prompt_template = f"""
        Extract the user_destination (city, country), points_of_interest (list of places), and user_interests from the User_Request.

        Do not assume or generate any values that are not explicitly stated in the text.

        User_Request: {user_input}

        ```json
        {{
            "user_destination": "...",
            "points_of_interest": [...],
            "user_interests": "..."
        }}
        ```
    """
    response = model.generate_content(prompt_template)

    try:
        cleaned_response = repair_json(response.text)
    except ValueError as e:
        print(f"Invalid JSON output from the model output: {response.text}")
        logging.info(f"ValueError in parse_input: {e}")
        cleaned_response = {"user_destination": None, "points_of_interest": [
        ], "user_interests": None}  # Or handle as needed

    cleaned_response = json.loads(cleaned_response)
    # print(cleaned_response)
    return cleaned_response


@tool
def hotel_search(user_destination: str, pois: list[str]) -> json:
    """
    Find a hotel near the user's points_of_interest in Florence, Italy

    Args:
        user_destination (str): The user's requested travel destination
        pois (list[str]): A string containing a comma-separated list of points_of_interest

    Returns:
       json: A JSON object with the name, description, and address of the recommended hotel
    """
    client = bigquery.Client()
    if user_destination == "Florence, Italy":
        print("checking BQ for hotel")
        poi_vector = get_text_embeddings(pois)
        vector_str = ', '.join(map(str, poi_vector))

        bigframes.pandas.close_session()
        bpd.options.bigquery.project = project_id
        bpd.options.bigquery.location = location

        vector_search_options = '\'{"use_brute_force":true}\''
        hotel_search_query = f'''
            CREATE TEMP TABLE hold AS
            SELECT
                array(select cast(elem as float64) from unnest(split(TRIM("{vector_str}", ", "), ",")) elem) AS poi_vector;

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
        hotel_json = bpd.read_gbq(
            hotel_search_query).to_dict(orient="records")[0]
        logging.info(hotel_json)
        print(hotel_json)
    else:
        print("user destination != Florence, Italy")
        hotel_json = None
    return hotel_json


@tool
def image_search_attractions(user_destination: str, pois: list[str]) -> json:
    """
    Find images of the user's points_of_interest in Florence, Italy

    Args:
        user_destination (str) : The user's requested travel destination
        pois (list[str]): An array of strings containing a comma-separated list of points_of_interest

    Returns:
        json: A JSON object with a GCS URI for an image of the user's points_of_interest
    """
    client = bigquery.Client()
    if user_destination == "Florence, Italy":
        print("checking BQ for images")
        poi_vector = get_multimodal_embeddings(pois)
        vector_str = ', '.join(map(str, poi_vector))

        bigframes.pandas.close_session()
        bpd.options.bigquery.project = project_id
        bpd.options.bigquery.location = location

        vector_search_options = '\'{"use_brute_force":true}\''
        attraction_image_query = f'''
            CREATE TEMP TABLE hold AS
            SELECT
                array(select cast(elem as float64) from unnest(split(TRIM("{vector_str}", ", "), ",")) elem) AS poi_vector;

            with search AS (
                SELECT
                    query.uri AS poi_uri
                FROM
                    VECTOR_SEARCH( TABLE hold,
                        'poi_vector',
                        TABLE `object_tables.image_embeds`,
                        'image_embedding_multimodal',
                        top_k => 5,
                        distance_type => 'COSINE',
                        OPTIONS => {vector_search_options})
                ORDER BY
                    distance
                    )

        SELECT poi_uri FROM search LIMIT 1
        '''
        attraction_image_dict = bpd.read_gbq(
            attraction_image_query).to_dict(orient="records")[0]
        attraction_image = attraction_image_dict["poi_uri"]
        print(attraction_image)
        if attraction_image not in ("", None):
            log_text = f"Attraction image found. URI: {attraction_image}"
            logging.info(log_text)
            print(log_text)
        else:
            print("user destination != Florence, Italy")
            attraction_image = None
    return attraction_image


@tool
def doc_search_attractions(user_destination: str, pois: list[str]) -> json:
    """
    Find suggested activities and experiences near the user's points_of_interest in Florence, Italy

    Args:
        user_destination str: The user's requested travel destination
        pois list[str]: An array of strings containing a comma-separated list of points_of_interest

    Returns:
        json: A JSON object with a string describing popular activities and experiences near the user's points_of_interest
    """
    client = bigquery.Client()
    if user_destination == "Florence, Italy":
        print("checking BQ for images")
        poi_vector = get_multimodal_embeddings(pois)
        vector_str = ', '.join(map(str, poi_vector))

        bigframes.pandas.close_session()
        bpd.options.bigquery.project = project_id
        bpd.options.bigquery.location = "US"

        vector_search_options = '\'{"use_brute_force":true}\''
        doc_search_query = f'''
            CREATE TEMP TABLE hold AS
            SELECT
                array(select cast(elem as float64) from unnest(split(TRIM("{vector_str}", ", "), ",")) elem) AS poi_vector;

            with search AS (
                SELECT
                    query.content AS content, distance
                FROM
                    VECTOR_SEARCH( TABLE hold,
                        'poi_vector',
                        TABLE `object_tables_us.doc_parsed_embed`,
                        'doc_embed_multimodal',
                        top_k => 5,
                        distance_type => 'COSINE',
                        OPTIONS => {vector_search_options})
                ORDER BY
                    distance
            ),

            subset AS(
                SELECT
                    *
                FROM
                    search
                ORDER BY
                    distance LIMIT 3
            )


        SELECT STRING_AGG(content) AS content FROM subset
        '''
        doc_search_dict = bpd.read_gbq(
            doc_search_query).to_dict(orient="records")[0]
        doc_search = doc_search_dict["content"]
        print(doc_search)
        if doc_search not in ("", None):
            log_text = "Doc text found found"
            logging.info(log_text)
            print(log_text)
        else:
            print("user destination != Florence, Italy")
            doc_search = None
    return doc_search


@tool
def weather_check(user_destination: str) -> json:
    """
    Find current weather conditions for a given destination.

    Args:
        user_destination: The user's requested destination in the format "City, Country".

    Returns:
        A JSON object with temperature, precipitation_probability, and wind_speed
    """
    gmaps = googlemaps.Client(key=get_gmaps_key("maps_api_key"))
    geocode_result = gmaps.geocode(user_destination)
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lon = geocode_result[0]["geometry"]["location"]["lng"]
    print("Destination: {user_destination}. Lat/lon: {lat}, {lon}")

    url = f"https://api.open-meteo.com/v1/forecast"
    # Set up the parameters for the API request
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "precipitation", "wind_speed_10m"],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch"
    }

    response_json = requests.get(url, params=params).json()
    current = response_json['current']
    print(current)
    current_weather_dict = {
        "temperature": current['temperature_2m'],
        "precipitation": current['precipitation'],
        "wind_speed": current['wind_speed_10m']
    }
    print("current weather in {user_destination} is: {current_weather_dict}")
    return current_weather_dict


parse_input_func = FunctionDeclaration(
    name="parse_input",
    description="Determine the destination and points of interest the user has identified",
    parameters={
        "type": "object",
        "properties": {
            "user_destination": {"type": "string", "description": "Destination in the format City, Country the user wants to plan a trip for"},
            "points_of_interest": {"type": "array", "description": "List of points of interest the user wants to see"},
            "user_interests": {"type": "string", "description": "User's interests for activities and experiences"},
        },
        "required": ["user_destination", "points_of_interest"],
    },
)

hotel_search_func = FunctionDeclaration(
    name="hotel_search",
    description="""
    Find a hotel near the user's points_of_interest in Florence, Italy

    Args:
        user_destination: The user's requested travel destination
        points_of_interest: An array of strings containing a comma-separated list of points_of_interest

    Returns:
        A JSON object with the name, description, and address of the recommended hotel
    """,
    parameters={
        "type": "object",
        "properties": {
            "user_destination": {"type": "string", "description": "Destination in the format City, Country the user wants to plan a trip for"},
            "points_of_interest": {"type": "array", "description": "List of points of interest the user wants to see"},
        },
        "required": ["user_destination", "points_of_interest"],
    },
)

image_search_attractions_func = FunctionDeclaration(
    name="image_search_attractions",
    description="""
    Find images of the user's points_of_interest in Florence, Italy

    Args:
        user_destination: The user's requested travel destination
        points_of_interest: An array of strings containing a comma-separated list of points_of_interest

    Returns:
        A JSON object with a GCS URI for an image of the user's points_of_interest
    """,
    parameters={
        "type": "object",
        "properties": {
            "user_destination": {"type": "string", "description": "Destination in the format City, Country the user wants to plan a trip for"},
            "points_of_interest": {"type": "array", "description": "List of points of interest the user wants to see"},
        },
        "required": ["user_destination", "points_of_interest"],
    },
)

doc_search_attractions_func = FunctionDeclaration(
    name="doc_search_attractions",
    description="""
    Find suggested activities and experiences near the user's points_of_interest in Florence, Italy

    Args:
        user_destination: The user's requested travel destination
        pois: An array of strings containing a comma-separated list of points_of_interest

    Returns:
        A JSON object with a string describing popular activities and experiences near the user's points_of_interest
    """,
    parameters={
        "type": "object",
        "properties": {
            "user_destination": {"type": "string", "description": "Destination in the format City, Country the user wants to plan a trip for"},
            "points_of_interest": {"type": "array", "description": "List of points of interest the user wants to see"},
        },
        "required": ["user_destination", "points_of_interest"],
    },
)

weather_check_func = FunctionDeclaration(
    name="weather_check",
    description="""
    Find current weather conditions for a given destination.

    Args:
        user_destination: The user's requested destination in the format "City, Country".

    Returns:
        A JSON object of the current weather, with temperature in fahrenheit, precipitation in inches, and wind_speed in mph
    """,
    parameters={
        "type": "object",
        "properties": {
            "user_destination": {"type": "string", "description": "Destination in the format City, Country the user wants to plan a trip for"},
        },
        "required": ["user_destination"],
    },
)

florence_tool_list = [
    parse_input,
    hotel_search,
    # image_search_attractions,
    doc_search_attractions,
    weather_check
]

florence_gemini_tool_list = [
    parse_input_func,
    hotel_search_func,
    image_search_attractions_func,
    doc_search_attractions_func,
    weather_check_func
]

florence_gemini_tools = Tool(
    function_declarations=florence_gemini_tool_list
)

# weather_tool_list = [
#     parse_input_func,
#     weather_check_func,
# ]

# weather_tools = Tool(
#     function_declarations=weather_tool_list
# )

all_tools = []
for tool in florence_tool_list:
    all_tools.append(tool)
# for tool in weather_tool_list:
#     all_tools.append(tool)


# florence_tool_node = ToolNode(florence_tool_list)
# weather_tool_node = ToolNode(weather_tool_list)
