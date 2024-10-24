from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from model_mgmt import config, instructions, prompt, toolkit
from typing import Annotated, TypedDict
from vertexai.generative_models import Content, Part, ChatSession

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


def get_chat_response(
        input: str,
        chat_session: ChatSession,
        function_response: bool = False,
) -> str:
    if function_response is True:
        contents = input
    else:
        complete_prompt = prompt.generate_template(input)
        contents = Content(role="user", parts=[
                           Part.from_text(text=complete_prompt)])

    response = chat_session.send_message(contents, stream=False)
    return response


def parse_preferences_func(
    input_text,  # Original user input
    func_output,
    func_response,
    chat_session: ChatSession,
):
    print(func_output)
    User_Email = func_output["User_Email"]
    full_text_input = prompt.generate_template(input_text)
    if User_Email in (' ', '', None):
        # Proceed with original input
        contents = Content(parts=[Part.from_text(full_text_input)])
    else:
        user_preferences = toolkit.get_user_preferences(user_email=User_Email)
        user_preferences_json = json.loads(user_preferences)
        function_response_part = Part.from_function_response(
            name="get_user_info",
            response={
                "content": user_preferences_json,
            }
        )
        contents = [
            function_response_part
        ]
    response = get_chat_response(
        contents, chat_session, function_response=True)
    return response.candidates[0]


def function_coordination(input: str, chat_session: ChatSession) -> str:
    input = input.strip()
    output = get_chat_response(input, chat_session)

    if isinstance(output.candidates, list):
        output_candidate = output.candidates[0]

        for function_call in output_candidate.function_calls:
            if function_call.name == "get_user_preferences":
                print("get_user_preferences function call found")
                func_output = function_call.args
                response = parse_preferences_func(
                    input, func_output, output_candidate.content, chat_session
                )
                return response.text

    # Default to returning the text if no get_user_preferences call
    return output_candidate.text
