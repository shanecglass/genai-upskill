import app.testing as testing
from google.cloud import aiplatform, bigquery
from json_repair import repair_json
import jsonschema
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
)

import bigframes
import bigframes.pandas as bpd
import vertexai

# flake8: noqa --E501[[]]

project_id = testing.project_id
location = testing.location

gen_config = {
    "temperature": 0.0,
    "top_p": .1,
    "top_k": 1,
    "candidate_count": 1,
    "max_output_tokens": 8000,
}

generation_config = GenerationConfig(
    temperature=gen_config['temperature'],
    # top_p=gen_config['top_p'],
    # top_k=gen_config['top_k'],
    candidate_count=gen_config['candidate_count'],
    # max_output_tokens=gen_config['max_output_tokens'],
)

system_instructions = """
You are a data generator, generating a set of data that can be used as a load into a database. For your datasets, apply the following schema, provided in JSON:

```
{"user_name": value, \\ the user name of the fictional user
"user_budget_range": value, \\ the fictional user's budget preference for trips on a scale of 1 through 5
"travel companions": value, \\ whether the fictional user typically travels solo, as a couple, with family, or with friends
"user_location": value \\ the fictional user's home city and state within the US
"destination_types: value \\ references for specific types of destinations such as beaches, cities, mountains, historical sites, nature
"trip_length": value \\ typical or preferred trip duration in number of days between 3 and 14
}
```

Provide the data in JSON format with no markdown syntax included. Validate you do not have duplicates. Unless specified otherwise, use the maximal output of 32,000 characters. Provide as many users as possible.
"""


model = GenerativeModel(
    model_name="gemini-1.5-pro-002",
    generation_config=generation_config,
    # safety_settings=safety_settings,
    system_instruction=system_instructions,
)

gen_prompt = """
    users for an online travel agency from across the United States using the maximal output of 32,000 characters
"""

output_schema = {
    "type": "object",
    "properties": {
        "user_name": {"type": "string"},
        "user_budget_range": {"type": "integer"},
        "travel_companions": {"type": "string"},
        "user_location": {"type": "string"},
        "destination_types": {"type": "string"},
        "trip_length": {"type": "integer"}
    },
    "required": ["user_name", "user_budget_range", "travel_companions", "user_location", "destination_types", "trip_length"]
}


def generate_users(prompt, project_id, location):
    vertexai.init(project=project_id, location=location)
    output = model.generate_content(prompt)
    json_output = repair_json(output.text)
    return json_output


json_output = generate_users(gen_prompt, project_id, location)


def load_to_bq(json_output, project_id, location):
    bpd.options.bigquery.project = project_id
    bpd.options.bigquery.location = location
    client = bigquery.Client(project=project_id)
    dataset_id = "travel_chat"
    dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
    dataset.location = location
    dataset = client.create_dataset(dataset, exists_ok=True)
    print(f"Created dataset {dataset.dataset_id} in {dataset.location}.")

    bq_df = bpd.read_json(json_output)
    # print(bq_df.head())
    table_id = f"{project_id}.{dataset_id}.users"
    bq_df.to_gbq(table_id, if_exists="replace", index=False)
    print(f"Loaded {len(bq_df)} rows into {table_id}.")


load_to_bq(json_output, project_id, location)
