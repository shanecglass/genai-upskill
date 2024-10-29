# flake8: noqa --E501


import testing
import vertexai
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Tool,
)

import os
from pathlib import Path
import shutil
import magika
import requests

m = magika.Magika()

vertexai.init(project=testing.project_id, location=testing.location)

MODEL_ID = "gemini-1.5-pro-002"
model = GenerativeModel(
    MODEL_ID,
    system_instruction=[
        "You are a coding expert.",
        "Your mission is to answer all code related questions with given context and instructions.",
    ],
)


def extract_code(repo_dir):
    """Create an index, extract content of code/text files."""

    code_index = []
    code_text = ""
    for root, _, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_dir)
            code_index.append(relative_path)

            file_type = m.identify_path(Path(file_path))
            if file_type.output.group in ("text", "code"):
                try:
                    with open(file_path) as f:
                        code_text += f"----- File: {relative_path} -----\n"
                        code_text += f.read()
                        code_text += "\n-------------------------\n"
                except Exception:
                    pass

    return code_index, code_text


repo_dir = "./app"

code_index, code_text = extract_code(repo_dir)


def get_code_prompt(question):
    """Generates a prompt to a code related question."""

    prompt = f"""
    Questions: {question}

    Context:
    - The entire codebase is provided below.
    - Here is an index of all of the files in the codebase:
      \n\n{code_index}\n\n.
    - Then each of the files is concatenated together. You will find all of the code you need:
      \n\n{code_text}\n\n

    Answer:
  """
    return prompt


question = """
Explain and troubleshoot this error: Cannot get the Candidate text.
Response candidate content part has no text.
Part:
{
  "function_call": {
    "name": "parse_input",
    "args": {
      "user_destination": "Florence, Italy"
    }
  }
}
Candidate:
{
  "content": {
    "role": "model",
    "parts": [
      {
        "function_call": {
          "name": "parse_input",
          "args": {
            "user_destination": "Florence, Italy"
          }
        }
      }
    ]
  },
  "finish_reason": "STOP",
  "avg_logprobs": -0.07077681356006199
}

Traceback
/python3.12/site-packages/mesop/server/server.py:179 | generate_data
 for _ in result:
/python3.12/site-packages/mesop/runtime/context.py:299 | run_event_handler
 yield from result
/Users/shanecglass/Documents/GitHub/genai-upskill/app/main.py:151 | on_click_submit_chat_msg
     start_time = time.time()
     output_message = respond_to_chat(input, state.output)
     assistant_message = ChatMessage(role=_ROLE_ASSISTANT)
     output.append(assistant_message)
     state.output = output
/Users/shanecglass/Documents/GitHub/genai-upskill/app/main.py:194 | respond_to_chat
         return result
     else:
         result = function_coordination(input)
         return result
 # Constants
/Users/shanecglass/Documents/GitHub/genai-upskill/app/model_calls.py:146 | function_coordination
 return response.candidates[0].text
/python3.12/site-packages/vertexai/generative_models/_generative_models.py:2298 | text
 raise ValueError(

"""

prompt = get_code_prompt(question)
contents = [prompt]

response = model.generate_content(contents)
print(f"\nAnswer:\n{response.text}")
