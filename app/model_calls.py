from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from model_mgmt import config, instructions, prompt, toolkit

from json_repair import repair_json
from vertexai.generative_models import Content, Part, ChatSession

import logging
import json


project_id = config.project_id
location = config.location
project_number = config.project_number

model_to_call = config.generative_model
endpoint_id = config.endpoint_id


# flake8: noqa --E501


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


def get_chat_response(
        input: str,
        chat_session: ChatSession,
        function_response: bool = False,
) -> str:

    if function_response is True:
        complete_prompt = input
        contents = complete_prompt
    else:
        complete_prompt = prompt.generate_template(input)
        contents = [complete_prompt]

    response = chat_session.send_message(contents, stream=False)
    return response


# def extract_from_input(func_output, func_response):
#     text_response = parse_hotel_func(
#         func_output, func_response)
#     return text_response

def parse_hotel_func(
    input_text,  # Original user input
    func_output,
    chat_session: ChatSession,
):
    # Extract arguments from the initial function call (if present).
    try:
       print(func_output)
       user_destination = func_output["User_Destination"]
       points_of_interest = func_output["Points_of_Interest"]

    # No function call yet. Extract Destination & POI from text
    except (IndexError, json.JSONDecodeError):
       # Add this function to toolkit.py
       parsed = toolkit.parse_input(user_input=input_text)
       user_destination = parsed["User_Destination"]
       points_of_interest = parsed["Points_of_Interest"]
       print(user_destination)
       print(points_of_interest)

    if user_destination == "Florence, Italy":
        hotel_details_json = toolkit.hotel_search(pois=points_of_interest)
        hotel_details = json.loads(hotel_details_json)

        function_response_part = Part.from_function_response(
            name="hotel_search",
            response=hotel_details
        )
        complete_prompt = Part.from_text(prompt.generate_template(input_text))
        contents = Content(role="user", parts=[
            complete_prompt, function_response_part])
    else:
        # Proceed with original input
        contents = Content(parts=[Part.from_text(input_text)])

    response = get_chat_response(
        contents, chat_session=chat_session, function_response=True)
    return response.candidates[0]


def function_coordination(input: str, chat_session: ChatSession) -> str:
    full_input = prompt.generate_template(input).strip()
    user_input_content = Content(
        role="user", parts=[Part.from_text(text=full_input)])
    output = chat_session.send_message(user_input_content, stream=False)

    if isinstance(output.candidates, list):
        output_candidate = output.candidates[0]

        for function_call in output_candidate.function_calls:
            if function_call.name == "hotel_search":
                print("hotel_search function call found")
                func_output = function_call.args
                response = parse_hotel_func(
                    full_input, func_output, output_candidate.content, chat_session=chat_session
                )
                return response.text

    return output_candidate.text  # Default to returning the text if no hotel_search call
