import logging
import os
import model_mgmt.primary as primary
import model_mgmt.instructions as instructions
import model_mgmt.toolkit as toolkit
import model_mgmt.testing as testing


from google.cloud import aiplatform
from langchain_google_vertexai import ChatVertexAI

from vertexai.generative_models import (
    ChatSession,
    GenerationConfig,
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Tool,

)
from enum import Enum

import json
import vertexai

# flake8: noqa --E501

# project_number = os.environ.get("PROJECT_NUMBER")
# gemma_endpoint_id = os.environ.get("GEMMA_ENDPOINT_ID")
# gemini_tuned_endpoint_id = os.environ.get("GEMINI_TUNED_ENDPOINT_ID")
# project_id = os.environ.get("PROJECT_ID")
# location = os.environ.get("LOCATION")

project_number = testing.project_number
gemma_endpoint_id = testing.gemma_endpoint_id
gemini_tuned_endpoint_id = testing.gemini_tuned_endpoint_id
project_id = testing.project_id
location = testing.location

configured_model = primary.configured_model
system_instructions = instructions.system_instructions

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


gen_config = {
    "temperature": 0.1,
    "top_p": .3,
    "top_k": 1,
    "candidate_count": 1,
    "max_output_tokens": 1048,
}


generation_config = GenerationConfig(
    temperature=gen_config['temperature'],
    top_p=gen_config['top_p'],
    top_k=gen_config['top_k'],
    candidate_count=gen_config['candidate_count'],
    max_output_tokens=gen_config['max_output_tokens'],
)


safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}


def model_to_call(Selected_Model=Selected_Model):
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
        if Selected_Model == Valid_Models['GEMINI_FLASH']:
            model_id = Selected_Model
        else:
            model_id = f"projects/{project_number}/locations/{location}/endpoints/{gemini_tuned_endpoint_id}"  # noqa --E501
        model_to_call = GenerativeModel(
            model_name=model_id,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=instructions.system_instructions,
            tools=[toolkit.tool_list]
        )
        endpoint_id = gemini_tuned_endpoint_id
    return model_to_call, endpoint_id, model_id


generative_model = model_to_call(Selected_Model)[0]
endpoint_id = model_to_call(Selected_Model)[1]
model_name = model_to_call(Selected_Model)[2]


def start_chat():
    vertexai.init(project=project_id, location=location)
    # chat_history = []
    # for h in history:
    #     message_content = Content(
    #         role=h.role, parts=[Part.from_text(h.content)])
    #     chat_history.append(message_content)
    chat_session = ChatVertexAI(
        model=model_name,
        temperature=gen_config['temperature'],
        top_p=gen_config['top_p'],
        top_k=gen_config['top_k'],
        candidate_count=gen_config['candidate_count'],
        max_output_tokens=gen_config['max_output_tokens'],
        safety_settings=safety_settings)
    # chat_session.invoke(system_message)
    return chat_session




# print(system_message)

chat_session = start_chat()
