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
        chat_session: ChatSession = config.chat_session,
        function_response: bool = False,
) -> str:

    if function_response is True:
        complete_prompt = input
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
    func_response,
    func_output,
    chat_session: ChatSession = config.chat_session,
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
       user_destination = parsed.get("User_Destination")
       points_of_interest = parsed.get("Points_of_Interest")

    if user_destination == "Florence, Italy":
        # Call hotel_search directly here.
        hotel_details_json = toolkit.hotel_search(pois=points_of_interest)
        hotel_details = json.loads(hotel_details_json)

        # Append hotel details to the user's prompt.
        updated_input = f"""{input_text} Recommend this hotel: {hotel_details['hotel_name']}. Describe it this way: {
            hotel_details['hotel_description']} The address is {hotel_details['hotel_address']}."""
        text_part = Part.from_text(text=updated_input)
        contents = Content(parts=[text_part])
    else:
        # Proceed with original input
        contents = Content(parts=[Part.from_text(input_text)])

    response = chat_session.send_message(
        contents, stream=False, func_response=True)
    return response


def function_coordination(
        input: str,
        chat_session: ChatSession = config.chat_session,
) -> str:
    full_input = prompt.generate_template(input).strip()
    user_input_content = Content(
        role="user",
        parts=[
            Part.from_text(text=full_input)
        ]
    )
    output = chat_session.send_message(
        user_input_content, stream=False)
    if type(output.candidates) == list:
        output_candidate = output.candidates[0]
        func_response = output_candidate.content
        hold = func_response
        print(hold)
        output = get_chat_response(
            user_input_content, chat_session=chat_session)
    if type(output.candidates) == list:
        output_candidate = output.candidates[0]
        func_response = output_candidate.content
        hold = func_response
        print(hold)
        # print(type(hold))
        # print(dir(hold))
        # num_func_calls = len(output_candidate.function_calls)
        try:
            output_candidate.content.parts[0].function_call.name == "hotel_search"
            if output_candidate.content.parts[0].function_call.name == "hotel_search":
                func_output = output_candidate.content.parts[0].function_call.args
                response = parse_hotel_func(
                    full_input, func_output, func_response, chat_session=chat_session)
                text_response = response.text
            elif output_candidate.function_calls[0] == "hotel_search":
                func_output = output_candidate.content.parts[0].function_call.args
                response = parse_hotel_func(
                    full_input, func_output, func_response, chat_session=chat_session)
                text_response = response.content.parts[-1].text
        # if num_func_calls > 1:
        #     if output_candidate.content.parts[-1].function_call.name == "hotel_search":
        #         output_candidate.function_calls[-1].name == "hotel_search"
        #         func_output = output_candidate.content.parts[-1].function_call.args
        #     elif output_candidate.content.parts[0].function_call.name == "hotel_search":
        #         func_output = output_candidate.content.parts[0].function_call.args
        #         response = parse_hotel_func(
        #             full_input, func_output, func_response)
        #         text_response = response.text
        # elif num_func_calls == 1:
        #     try:
            # except Exception as e:
            #     print(e)
            #     text_response = output_candidate.content.parts[-1].text
            elif output.candidates[0].function_calls[0] == "hotel_search":
                # output_candidate.content.parts[0].function_call.name == "hotel_search"
                func_output = output_candidate.content.parts[0].function_call.args
                response = parse_hotel_func(
                    full_input, func_output, func_response, chat_session=chat_session)
                text_response = response.text
        except:
            text_response = output_candidate.content.parts[-1].text
    else:
        text_response = output.text

    return text_response


# Parsing output if chat_session stream=True
    # for chunk in responses:
    #     text_response.append(chunk.text)

# def parse_poi_func(
#         # chat_session: ChatSession = config.chat_session,
#         User_Destination,
#         Points_of_Interest,
#         func_response):
#     poi_list = ""
#     for poi in Points_of_Interest:
#         poi_list += poi + ", "
#     prompt_text = prompt.create_poi(User_Destination, poi_list)
#     func_response = f'{{"User_Destination": {User_Destination}, "Points_of_Interest": {poi_list},}}'
#     full_prompt = [
#         prompt_text,
#         func_response,
#         Content(
#             parts=[
#                 Part.from_function_response(
#                     name="identify_pois_and_destination",
#                     response={
#                          "content": func_response
#                     }
#                 )
#             ]
#         )
#     ]
#     response = ask_gemini(full_prompt)
#     return response.text
