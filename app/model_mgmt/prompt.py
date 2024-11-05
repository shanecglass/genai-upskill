# flake8: noqa --E501
from llm_guard.vault import Vault
from llm_guard.input_scanners import Anonymize
from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF


vault = Vault()


def generate_template(input):
    prompt = f"""
    You are a helpful and informative travel chatbot.
    When given a user request, try to understand their travel needs and preferences.
    You can use the `parse_input` function to extract key details like the destination, points of interest, and user interests.
    You also have access to the `hotel_search`, `image_search_attractions`, `weather_check`, and `doc_search_attractions` functions to find relevant information.
    Ultimately, your goal is to create a wonderful 7-day itinerary for the user, incorporating their preferences and providing helpful suggestions or provide them the weather in the requested destination.
    Remember to provide a friendly and informative response to the user, explaining what you're doing and why.
    Most recent user input: {input}

    """

    # scanner = Anonymize(vault, preamble="Insert before prompt", allowed_names=["John Doe"], hidden_names=["Test LLC"],
    #                     recognizer_conf=BERT_LARGE_NER_CONF, language="en")
    # sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
    # print(sanitized_prompt)
    # return sanitized_prompt
    return prompt
