import bigframes.pandas as bpd
import os
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    grounding,
    Tool
)

from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from json_repair import repair_json

# flake8: noqa --E501

project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("LOCATION")


bpd.options.bigquery.project = project_id
bpd.options.bigquery.location = location

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


@tool
def identify_pois_and_destination(user_input: str):
    """Determine the User_Destination and Points_of_Interest the user has identified in the chat"""
    prompt = f"""
    Your sole purpose is to identify key information in a user's request.
    A user is asking for travel information to a specific location and points of interest.
    Parse the user's input to identify the following:
        * User_Destination (city or country they want to visit)
        * Points of interest (specific places they want to see)

    User input: {user_input}

    Return the information in JSON format with the following structure:
    {{
        "User_Destination": "destination_city_or_country",
        "Points_of_Interest": ["point_of_interest_1", "point_of_interest_2"]
    }}
    """

    response = model.predict(prompt)
    try:
        cleaned_response = repair_json(response.text)
    except ValueError as e:
        print(f"Invalid JSON output from the model: {response.text}")
        # Return a default empty result or handle the error appropriately
        cleaned_response = {"User_Destination": None, "Points_of_Interest": []}
    return cleaned_response


@tool
def hotel_search(travel_info):
    """Search the database of hotels to find a hotel near the specified Points_of_Interest in Florence from the identify_pois_and_destination function"""
    hotel_search_query = f"""
        CREATE TEMP TABLE hold AS
        SELECT
        ml_generate_embedding_result AS test
        FROM
        ML.GENERATE_EMBEDDING( MODEL `model_fine_tuning.text_embedding_004`,
            (
            SELECT
            "Show me hotels near {travel_info['Points_of_Interest']}" AS content),
            STRUCT(TRUE AS flatten_json_output) );

        with search AS (
            SELECT
                query.hotel_name AS name,
                query.hotel_address AS address,
                query.hotel_description AS description
            FROM
                VECTOR_SEARCH( TABLE hold,
                    'test',
                    TABLE `hotels.florence_embeddings`,
                    'nearest_attractions_embeddings',
                    top_k => 5,
                    distance_type => 'COSINE',
                    OPTIONS => '{"use_brute_force":true}')
            ORDER BY
                distance
                )
    SELECT * FROM search LIMIT 1
    """
    hotel_json = bpd.read_gbq(hotel_search_query).to_json()
    return hotel_json


identify_pois_and_destination_func = FunctionDeclaration(
    name="identify_pois_and_destination",
    description="Determine the User_Destination and Points_of_Interest the user has identified in the chat",
    parameters={
        "type": "object",
        "properties": {"user_input": {"type": "string", "description": "Input message from users"}},
    },
)

hotel_search_func = FunctionDeclaration(
    name="travel_info",
    description="Search the database of hotels to find a hotel near the specified Points_of_Interest in Florence",
    parameters={
        "type": "object",
        "properties": {"travel_info": {"type": "string", "description": "Input message from users"}},
    },
)

print(hotel_search.name)

tool_list = Tool(
    function_declarations=[
        identify_pois_and_destination_func,
        hotel_search_func
    ],
)

tools = [identify_pois_and_destination, hotel_search]
tool_node = ToolNode(tools)
