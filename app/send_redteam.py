from model_mgmt.prompt import generate_template
from model_calls import react_agent


def call_api(prompt: str, options, context):
    full_input = generate_template(prompt)
    try:
        response = react_agent.query(input=full_input)
        if response is None:
            return {
                "output": None,
                "error": "No response received"
            }
        else:
            return {
                "output": response
            }
    except Exception as e:
        return {
            "output": None,
            "error": f"An error occurred: {str(e)}"
        }
