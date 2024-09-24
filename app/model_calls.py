from model_mgmt import config, prompt
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

import vertexai

project_id = config.project_id
location = config.location
project_number = config.project_number
endpoint_id = config.endpoint_id
model_to_call = config.model_to_call

# flake8: noqa --E501


def ask_gemma(
    user_input: str,
):
    system_instructions = ""
    for instruction in config.system_instructions:
        system_instructions += instruction + " "
    complete_prompt = f"""
        Instructions: {system_instructions}

        {prompt.template}"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model"
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


def ask_gemini(user_input: str):
    vertexai.init(project=project_id, location=location)

    complete_prompt = f"""
        User input: {prompt.template}{user_input}
        Answer:
        """

    contents = [complete_prompt]
    response = model_to_call.generate_content(
        contents,
        generation_config=config.generation_config,
        safety_settings=config.safety_settings,
    )
    return response.text
