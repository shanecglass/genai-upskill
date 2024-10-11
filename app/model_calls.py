from model_mgmt import config, instructions, prompt
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

from vertexai.generative_models import ChatSession

project_id = config.project_id
location = config.location
project_number = config.project_number

model_configs = config.model_to_call(config.Selected_Model)
model_to_call = model_configs[0]
endpoint_id = model_configs[1]


# flake8: noqa --E501


def ask_gemma(
    history: list,
):
    full_input = ""
    for h in history:
        full_input += "<start_of_turn>{role}\n{content}<end_of_turn>\n".format(
            role=h.role, content=h.content)
    full_input += "<start_of_turn>model\n"
    system_instructions = ""
    for instruction in instructions.system_instructions:
        system_instructions += instruction + " "
    complete_prompt = f"""
        Instructions: {system_instructions}

        {prompt.template}"<start_of_turn>user\n{full_input}<end_of_turn>\n<start_of_turn>model"
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
#         history: list,
# ):
#     chat_history = "\n".join(message.content for message in history)
#     # full_input = f"{chat_history}\n{input}"
#     vertexai.init(project=project_id, location=location)
#     complete_prompt = f"""
#         User input: {prompt.template}{input}
#         Answer:
#         """
#     contents = [complete_prompt]
#     response = model_to_call.send_message(
#         contents,
#         generation_config=config.generation_config,
#         safety_settings=config.safety_settings,
#     )
#     return response.text

def ask_gemini(
        chat_session: ChatSession,
        input_text: str,
) -> str:
    # input_text = input["messages"][-1].key()
    response = chat_session.send_message(input_text)
    response_text = response.text
    return response_text
