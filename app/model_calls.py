from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from model_mgmt import config, instructions, prompt, toolkit
from json_repair import repair_json
from vertexai.generative_models import Content, Part

import logging


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
    complete_prompt = f"""
        Instructions: {system_instructions}

        {prompt.template}"<start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model"
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


def ask_gemini(
        input: str,
        function_response: bool = False,
):
    # vertexai.init(project=project_id, location=location)
    if function_response is True:
        complete_prompt = input
    else:
        complete_prompt = prompt.generate_template(input)

    contents = [complete_prompt]
    response = model_to_call.generate_content(contents)
    return response

# def get_chat_response(
#         chat_session: ChatSession,
#         prompt: str) -> str:
#     output = chat_session.send_message(prompt, stream=False)
#     return output

def extract_from_input(func_output, output_candidate):
    User_Destination = func_output.args["User_Destination"]
    Points_of_Interest = func_output.args["Points_of_Interest"]
    User_Interests = func_output.args["User_Interests"]
    func_response = output_candidate.content
    text_response = parse_hotel_func(
        User_Destination, User_Interests, Points_of_Interest, func_response)
    return text_response

def parse_hotel_func(
        # chat_session: ChatSession,
        User_Destination: str,
        User_Interest,
        Points_of_Interest: list,
        func_response
        ):
    if User_Destination == "Florence, Italy":
        recommended_hotel = toolkit.hotel_search(Points_of_Interest)
    else:
        recommended_hotel = ""
    text_prompt = prompt.create_itinerary(
        User_Destination, User_Interest, Points_of_Interest, recommended_hotel)

    full_prompt = [
        text_prompt,
        func_response,
        Content(
            parts=[
                Part.from_function_response(
                    name="hotel_search",
                    response={
                        "content": recommended_hotel,
                    },
                )
            ],
        ),
    ]
    response = ask_gemini(full_prompt, function_response=True)
    return response.candidates[0].content.parts[0].text


def function_coordination(
        # chat_session: ChatSession,
        prompt: str
) -> str :
    output = ask_gemini(prompt)
    if type(output.candidates) == list:
        output_candidate = output.candidates[0]
        func_response = output_candidate.content
        try:
            func_output = output_candidate.content.parts[1].function_call.name
            if func_output in ("parse_input", "parse_input_func"):
                    text_response = extract_from_input(
                        func_response, output_candidate)
                    return text_response

            elif output_candidate.content.parts[0].function_call.name in ("hotel_search", "hotel_search_func"):
                text_response = output.text
                return text_response

            else:
                text_response = output.text
                return text_response

        except IndexError:
            try:
                func_output = output_candidate.content.parts[0].function_call.name
                if func_output in ("parse_input", "parse_input_func"):
                    text_response = extract_from_input(
                        func_response, output_candidate)
                    return text_response

                elif output_candidate.content.parts[0].function_call.name in ("hotel_search", "hotel_search_func"):
                    text_response = output.text
                    return text_response

                else:
                    text_response = output.text
                    return text_response

            except:
                text_response = output.text
                return text_response

        except AttributeError:
            text_response = output.text
            return text_response

    else:
        text_response = output.text
        return text_response


# Parsing output if chat_session stream=True
                # for chunk in responses:
                #     text_response.append(chunk.text)

# def parse_poi_func(
#         # chat_session: ChatSession,
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
