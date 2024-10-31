from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_core.prompts import ChatPromptTemplate
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from model_mgmt import config, instructions, prompt, toolkit

from json_repair import repair_json
from vertexai.generative_models import Content, GenerationResponse, Part, ChatSession
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

from langgraph.graph import StateGraph, START, END, MessageGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from vertexai.preview import reasoning_engines

from PIL import Image
# from IPython.display import Image, display


from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles



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
        public_url = poi_image_uri.replace(
            "gs://", "https://storage.googleapis.com/")
        function_response_part = Part.from_function_response(
            name="image_search_attractions",
            response={"content": public_url},
        )
        # complete_prompt = Part.from_text(input_text)
        output = function_response_part

    else:
        # Proceed with original input
        output = Part.from_text(input_text)
        public_url = None
    return output, public_url


def parse_weather_check(
    input_text,  # Original user input
    func_output,
    chat_session
):
    # Extract arguments from the initial function call (if present).
    try:
       print(func_output)
       user_destination = func_output["user_destination"]

    # No function call yet. Extract Destination & POI from text
    except (IndexError, json.JSONDecodeError):
       # Add this function to toolkit.py
       parsed = toolkit.parse_input(user_input=input_text)
       user_destination = parsed["user_destination"]
       print(user_destination)

    current_weather = toolkit.weather_check(
        user_destination=user_destination)

    function_response_part = Part.from_function_response(
        name="weather_check",
        response={"content": current_weather},
    )
    output = function_response_part

    return output


# TODO: Spend <4 hours converting this to use either:
# LangChain: https://www.googlecloudcommunity.com/gc/Community-Blogs/Building-and-Deploying-AI-Agents-with-LangChain-on-Vertex-AI/ba-p/748929
# Vertex Reasoning Agent: https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/intro_reasoning_engine.ipynb
# Maybe Langgraph but start from scratch: https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#react-agent
def function_coordination(input: str, chat_session) -> str:
    full_input = prompt.generate_template(input).strip()
    input_part = Part.from_text(text=full_input)
    user_input_content = Content(
        role="user", parts=[input_part])
    output = get_chat_response(user_input_content, chat_session)
    public_url = None

    if isinstance(output.candidates, list):
        output_candidate = output.candidates[0]
        if True in (output_candidate.function_calls is None, output_candidate.function_calls == []):
            text_response = output_candidate.text
        else:
            function_responses = []
            function_responses.append(input_part)
            for function_call in output_candidate.function_calls:
                if function_call.name == "hotel_search":
                    print("hotel_search function call found")
                    func_output = function_call.args
                    response = parse_hotel_func(
                        full_input, func_output, chat_session)
                    function_responses.append(response)
                elif function_call.name == "parse_input":
                    print("parse_input function call found")
                    func_output = function_call.args
                    response = parse_input_func(
                        full_input, func_output, chat_session)
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
                elif function_call.name == "weather_check":
                    print("weather_check function call found")
                    func_output = function_call.args
                    response = parse_weather_check(
                        full_input, func_output, chat_session)
                    function_responses.append(response)
            # Create Content with function_responses
            if function_responses:
                # Send function responses in a single turn
                function_response_content = Content(
                    parts=function_responses)
                response = get_chat_response(
                    function_response_content, chat_session=chat_session)
                text_response = response.candidates[0].text
            else:
                text_response = output_candidate.text
    else:
        text_response = output_candidate.text

    return text_response, public_url


# react_agent = create_react_agent(
#     model=config.agent_chat_session,
#     tools=toolkit.florence_tool_list
# )

prompt_template = {
    "user_input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
} | ChatPromptTemplate.from_messages([
    ("system", instructions.system_instructions),
    ("placeholder", "{history}"),
    ("user", "{user_input}"),
    ("placeholder", "{agent_scratchpad}"),
])


react_agent = reasoning_engines.LangchainAgent(
    model=config.model_name,
    prompt=prompt_template,
    tools=toolkit.florence_tool_list,
)

# img = react_agent.get_graph().draw_mermaid_png(
#     draw_method=MermaidDrawMethod.API,
# )
# with open("agent_graph.png", "wb") as f:
#     f.write(img)


# def router(state: MessagesState) -> str:
#     messages = state["messages"]
#     last_message = messages[-1]
#     if last_message.tool_calls:
#         return "tools"
#     else:
#         return END


# class LangGraphApp:
#     def __init__(self, chat_session, project_id: str = project_id, location: str = location) -> None:
#         self.project_id = project_id
#         self.location = location
#         self.chat_session = chat_session

#     # The set_up method is used to define application initialization logic
#     def set_up(self) -> None:

#         builder = MessageGraph()
#         model_with_tools = self.chat_session
#         builder.add_node("tools", model_with_tools)

#         florence_tool_node = toolkit.florence_tool_node
#         # weather_tool_node = toolkit.weather_tool_node
#         builder.add_node("florence_trip_planning", florence_tool_node)
#         # builder.add_node("weather_check", weather_tool_node)
#         builder.add_edge("florence_trip_planning", END)
#         # builder.add_edge("weather_check", END)

#         builder.set_entry_point("tools")
#         builder.add_conditional_edges("tools", router)

#         self.runnable = builder.compile()

#     # The query method will be used to send inputs to the agent
#     def query(self, input: str):
#         """Query the application.

#         Args:
#             input: The user message.

#         Returns:
#             str: The LLM response.
#         """
#         chat_history = self.runnable.invoke(
#             HumanMessage(prompt.generate_template(input)))

#         return chat_history[-1].content


# agent = LangGraphApp(config.agent_chat_session)
# agent.set


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
