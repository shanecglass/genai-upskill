from google.cloud import aiplatform
# from langchain_google_vertexai import ChatVertexAI
from vertexai.generative_models import (
    ChatSession,
    GenerationConfig,
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
)

import logging
import json
import model_mgmt.instructions as instructions
import model_mgmt.primary as primary
import model_mgmt.testing as testing
import model_mgmt.toolkit as toolkit
import vertexai

# flake8: noqa --E501


gen_config = {
    "temperature": 0.0,
    "top_p": .1,
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

model_to_call = GenerativeModel(
    model_name="gemini-1.5-flash-002",
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction=instructions.system_instructions,
    tools=[toolkit.tool_list]
)


def get_chat_response(
        # chat: ChatSession,
        prompt: str) -> str:
    chat_session = model_to_call.start_chat()
    responses = chat_session.send_message(prompt, stream=False)
    try:
        print(responses.candidates[0].content.parts[0].text)
    except:
        print("none")
    return (responses)


get_chat_response("let's go to florence.")

