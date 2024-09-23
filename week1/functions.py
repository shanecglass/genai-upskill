import os
import requests

from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)

project_number = os.environ.get("PROJECT_NUMBER")
endpoint_id = os.environ.get("ENDPOINT_ID")
location = os.environ.get("LOCATION")



destination_tool = Tool(
    function_declarations=[

    ],
)
