import logging
import os
import model_mgmt.primary as primary
import model_mgmt.instructions as instructions
TESTING = False

from google.cloud import aiplatform
from vertexai.generative_models import (
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
    GenerativeModel,
    Tool,
    grounding
)
from enum import Enum

import vertexai

# flake8: noqa --E501

project_number = os.environ.get("PROJECT_NUMBER")
gemma_endpoint_id = os.environ.get("GEMMA_ENDPOINT_ID")
gemini_tuned_endpoint_id = os.environ.get("GEMINI_TUNED_ENDPOINT_ID")
project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("LOCATION")

configured_model = primary.configured_model

Valid_Models = {
    # Define the list of valid models.
    # This is used to check if the selected model defined above is valid
    # Only change this if you have added support for a new model
    "GEMINI_FLASH": "gemini-1.5-flash-002",
    "GEMMA": "gemma2-9b-it",
    "GEMINI_PRO_TUNED": "gemini-1.0-pro-002_tuned",
}

# Sets a default value for the chat to use in case an invalid select is made
default_model = Valid_Models["GEMINI_FLASH"]


def model_check(
        ##
        # Check to see if the selected_model is a valid option.
        # If it is, the chat will use selected_model
        # If not, the chat will use default_model and log an error
        ##
            Valid_Models: dict = Valid_Models,
            configured_model: str = configured_model,
    default_model: str = default_model
) -> str:
    try:
        configured_model in Valid_Models.keys()
        Model = Valid_Models[configured_model]
        logging.info(f'Selected model for this chat is {Model}')
        print(f'Selected model for this chat is {Model}')
    except ValueError:
        logging.info(ValueError)
        Model = default_model
        logging.info(f'''
                     Invalid model selection in `primary.py` file.
                     Defaulting to {Model}
                    ''')
    return (Model)


Selected_Model = model_check()


def model_to_call(Selected_Model):
    if Selected_Model == Valid_Models['GEMMA']:
        model_id = gemma_endpoint_id
        api_endpoint = f"{location}-aiplatform.googleapis.com"
        # The AI Platform services require regional API endpoints.
        client_options = {"api_endpoint": api_endpoint}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        client = aiplatform.gapic.PredictionServiceClient(
            client_options=client_options)
        model_to_call = client.endpoint_path(
            project=project_number, location=location, endpoint=model_id
        )
        endpoint_id = gemma_endpoint_id
    else:
        vertexai.init(project=project_id, location=location)
        tools = [
            Tool.from_google_search_retrieval(
                google_search_retrieval=grounding.GoogleSearchRetrieval()
            ),
        ]
        if Selected_Model == Valid_Models['GEMINI_FLASH']:
            model_id = Selected_Model
        else:
            model_id = f"projects/{project_number}/locations/{location}/endpoints/{gemini_tuned_endpoint_id}"  # noqa --E501

        model_to_call = GenerativeModel(
            model_name=model_id,
            system_instruction=instructions.system_instructions,
            tools=tools
        )
        endpoint_id = gemini_tuned_endpoint_id
    return model_to_call, endpoint_id, tools

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=.3,
    top_k=1,
    candidate_count=1,
    max_output_tokens=1024,
)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}
