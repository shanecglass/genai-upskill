from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from model_mgmt import config, instructions, prompt, toolkit

from json_repair import repair_json
from vertexai.generative_models import Content, GenerationResponse, Part, ChatSession

import logging
from typing import Any
import json


project_id = config.project_id
location = config.location
project_number = config.project_number

model_to_call = config.generative_model
endpoint_id = config.endpoint_id


# flake8: noqa --E501

function_handler = {
    "hotel_search": toolkit.hotel_search,
    "parse_input": toolkit.parse_input,
    "image_search_attractions": toolkit.image_search_attractions,
}


def ask_gemma(
    input: list,
):

    system_instructions = ""
    for instruction in instructions.system_instructions:
        system_instructions += instruction + " "
    input = f"<start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model"
    full_text = prompt.generate_template(input)
    complete_prompt = f"""
        Instructions: {system_instructions}

        {full_text}
        """

    instances = {"prompt": complete_prompt,
                 "max_tokens": 512,
                 "temperature": 0.1,
                 "top_p": .3,
                 "top_k": 1,
                 }

    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())

    response = config.client.predict(
        endpoint=model_to_call, instances=instances, parameters=parameters
    )
    return (response.predictions[0])


def get_chat_response(
        input: str,
        chat_session,
        function_response: bool = False,
) -> str:
    if type(input) == Content:
        contents = input
    else:
        complete_prompt = prompt.generate_template(input)
        contents = Content(role="user", parts=[
                           Part.from_text(complete_prompt)])

    response = chat_session.send_message(contents, stream=False)
    print(chat_session.history)
    return response


def parse_input_func(
    input_text,  # Original user input
    func_output,
    chat_session
):
    try:
       print(func_output)
       user_destination = func_output["user_destination"]
       points_of_interest = func_output["points_of_interest"]
       parsed_json = func_output

    # No function call yet. Extract Destination & POI from text
    except (IndexError, json.JSONDecodeError):
       # Add this function to toolkit.py
       parsed_json = toolkit.parse_input(user_input=input_text)
       user_destination = parsed_json["user_destination"]
       points_of_interest = parsed_json["points_of_interest"]

    output = Part.from_function_response(
        name="parse_input", response={"content": parsed_json})

    return output


def parse_hotel_func(
    input_text,  # Original user input
    func_output,
    chat_session
):
    # Extract arguments from the initial function call (if present).
    try:
       print(func_output)
       user_destination = func_output["user_destination"]
       points_of_interest = func_output["points_of_interest"]

    # No function call yet. Extract Destination & POI from text
    except (IndexError, json.JSONDecodeError):
       # Add this function to toolkit.py
       parsed = toolkit.parse_input(user_input=input_text)
       user_destination = parsed["user_destination"]
       points_of_interest = parsed["points_of_interest"]
       print(user_destination)
       print(points_of_interest)

    if user_destination == "Florence, Italy":
        hotel_details = toolkit.hotel_search(
            user_destination=user_destination, pois=points_of_interest)

        function_response_part = Part.from_function_response(
            name="hotel_search",
            response={"content": hotel_details},
        )
        # complete_prompt = Part.from_text(input_text)
        output = function_response_part
    else:
        # Proceed with original input
        output = Part.from_text(input_text)
    return output

# Update to check for destination and POIs, then call function to get public_url from URI.
# Doublecheck function parameters are right


def parse_doc_attractions(
    input_text,  # Original user input
    func_output,
    chat_session
):
    # Extract arguments from the initial function call (if present).
    try:
       print(func_output)
       user_destination = func_output["user_destination"]
       points_of_interest = func_output["points_of_interest"]

    # No function call yet. Extract Destination & POI from text
    except (IndexError, json.JSONDecodeError):
       parsed = toolkit.parse_input(user_input=input_text)
       user_destination = parsed["user_destination"]
       points_of_interest = parsed["points_of_interest"]

    if user_destination == "Florence, Italy":
        activities = toolkit.doc_search_attractions(
            user_destination=user_destination, pois=points_of_interest)

        function_response_part = Part.from_function_response(
            name="doc_search_attractions",
            response={"content": activities},
        )
        # complete_prompt = Part.from_text(input_text)
        output = function_response_part
    else:
        # Proceed with original input
        output = Part.from_text(input_text)
    return output


def parse_attraction_images(
    input_text,  # Original user input
    func_output,
    chat_session
):
    # Extract arguments from the initial function call (if present).
    try:
       print(func_output)
       user_destination = func_output["user_destination"]
       points_of_interest = func_output["points_of_interest"]

    # No function call yet. Extract Destination & POI from text
    except (IndexError, json.JSONDecodeError):
       parsed = toolkit.parse_input(user_input=input_text)
       user_destination = parsed["user_destination"]
       points_of_interest = parsed["points_of_interest"]

    if user_destination == "Florence, Italy":
        poi_image_uri = toolkit.image_search_attractions(
            user_destination=user_destination, pois=points_of_interest)

        function_response_part = Part.from_function_response(
            name="image_search_attractions",
            response={"content": poi_image_uri},
        )
        # complete_prompt = Part.from_text(input_text)
        output = function_response_part
        public_url = poi_image_uri.replace(
            "gs://", "https://storage.googleapis.com/")
    else:
        # Proceed with original input
        output = Part.from_text(input_text)
        public_url = None
    return output, public_url


def function_coordination(input: str, chat_session) -> str:
    full_input = prompt.generate_template(input).strip()
    user_input_content = Content(
        role="user", parts=[Part.from_text(text=full_input)])
    output = get_chat_response(user_input_content, chat_session)
    public_url = None

    if isinstance(output.candidates, list):
        output_candidate = output.candidates[0]
        if True in (output_candidate.function_calls is None, output_candidate.function_calls == []):
            text_response = output_candidate.text
        else:
            function_responses = []
            function_responses.append(Part.from_text(text=full_input))
            for function_call in output_candidate.function_calls:
                if function_call.name == "hotel_search":
                    print("hotel_search function call found")
                    func_output = function_call.args
                    response = parse_hotel_func(
                        full_input, func_output, chat_session=chat_session
                    )
                    # Collect function response parts
                    function_responses.append(response)
                elif function_call.name == "parse_input":
                    print("parse_input function call found")
                    func_output = function_call.args
                    response = parse_input_func(
                        full_input, func_output, chat_session=chat_session)
                    # Collect function response parts
                    function_responses.append(response)
                elif function_call.name == "image_search_attractions":
                    print("image_search_attractions function call found")
                    func_output = function_call.args
                    response, public_url = parse_attraction_images(
                        full_input, func_output, chat_session)
                    function_responses.append(response)
                elif function_call.name == "doc_search_attractions":
                    print("doc_search_attractions function call found")
                    func_output = function_call.args
                    response = parse_doc_attractions(
                        full_input, func_output, chat_session)
                    function_responses.append(response)

            # Create Content with function_responses
            if function_responses:
                # Send function responses in a single turn
                function_response_content = Content(parts=function_responses)
                response = get_chat_response(
                    function_response_content, chat_session=chat_session)
                text_response = response.candidates[0].text
            else:
                text_response = output_candidate.text
    else:
        text_response = output_candidate.text

    return text_response, public_url


# def ask_gemini(
#         input: str,
#         function_response: bool = False,
# ):
#     # vertexai.init(project=project_id, location=location)
#     if function_response is True:
#         complete_prompt = input
#     else:
#         complete_prompt = prompt.generate_template(input)

#     contents = [complete_prompt]
#     response = model_to_call.generate_content(contents)
#     return response

# def extract_from_input(func_output, func_response):
#     text_response = parse_hotel_func(
#         func_output, func_response)
#     return text_response

# def parse_hotel_func(
#     input_text,  # Original user input
#     func_output,
# ):
#     # Extract arguments from the initial function call (if present).
#     try:
#        print(func_output)
#        user_destination = func_output["user_destination"]
#        points_of_interest = func_output["points_of_interest"]

#     # No function call yet. Extract Destination & POI from text
#     except (IndexError, json.JSONDecodeError):
#        # Add this function to toolkit.py

#        print(user_destination)
#        print(points_of_interest)

#     if user_destination == "Florence, Italy":
#         hotel_details_json = toolkit.hotel_search(pois=points_of_interest)
#         hotel_details = json.loads(hotel_details_json)

#         function_response_part = Part.from_function_response(
#             name="hotel_search",
#             response={
#                 "content": hotel_details}
#         )
#         complete_prompt = Part.from_text(prompt.generate_template(input_text))
#         contents = Content(role="user", parts=[
#             complete_prompt, function_response_part])
#     else:
#         # Proceed with original input
#         contents = Content(parts=[Part.from_text(input_text)])

#     response = get_chat_response(
#         contents, function_response=True)
#     return response.candidates[0]
