import json
import vertexai

from google.cloud import pubsub_v1
from model_mgmt import config
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

project_id = config.project_id
location = config.location
Selected_Model = config.Selected_Model


def tokenizer(Selected_Model):
    if Selected_Model == config.Valid_Models['GEMINI_FLASH']:
        tokenizer = config.model_to_call(config.Selected_Model[0])
    if Selected_Model == config.Valid_Models['GEMINI_PRO_TUNED']:
        vertexai.init(project=project_id, location=location)
        tokenizer = GenerativeModel("gemini-1.5-pro-002")
    else:
        tokenizer = None

    return (tokenizer)


tokenizer = tokenizer(Selected_Model)

message_pubsub_topic_id = f"projects/{project_id}/topics/chatbot_messages"
reply_pubsub_topic_id = f"projects/{project_id}/topics/chatbot_replies"

publisher = pubsub_v1.PublisherClient()
message_topic_path = publisher.topic_path(project_id, message_pubsub_topic_id)
reply_topic_path = publisher.topic_path(
    project_id, reply_pubsub_topic_id)


def get_text_embeddings(text_input):
    text_embed_model = TextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003")
    text_embeddings = text_embed_model.get_embeddings([text_input])
    text_embeddings = text_embed_model.get_embeddings([text_input])
    for embedding in text_embeddings:
        vector = embedding.values
        return vector


def publish_message_pubsub(
        message_text,
        submit_time,
        message_count,
        session_id):
    message_text = message_text.replace("\n", " ").replace(
        "  ", " ").replace("*", "").strip()
    message_embed = json.dumps(get_text_embeddings(message_text))
    if tokenizer is None:
        dict = {"message_text": message_text,
                "message_embedding": message_embed,
                "message_time": submit_time,
                "message_count": message_count,
                "session_id": session_id,
                "token_count": None,
                "total_billable_characters": None,
                }
    else:
        token_count = tokenizer.count_tokens(message_text)
        dict = {"message_text": message_text,
                "message_embedding": message_embed,
                "message_time": submit_time,
                "message_count": message_count,
                "session_id": session_id,
                "token_count": token_count.total_tokens,
                "total_billable_characters": token_count.total_billable_characters,  # noqa --E501
                }
    data_string = json.dumps(dict)
    data = data_string.encode("utf-8")
    future = publisher.publish(message_pubsub_topic_id, data)
    return (future)


def publish_reply_pubsub(
        reply_text,
        reply_time,
        reply_count,
        session_id,
        time_to_reply):
    reply_text = reply_text.replace("\n", " ").replace(
        "  ", " ").replace("*", "").strip()
    reply_embed = json.dumps(get_text_embeddings(reply_text))
    dict = {"reply_text": reply_text,
            "reply_embedding": reply_embed,
            "reply_time": reply_time,
            "reply_count": reply_count,
            "session_id": session_id,
            "response_time": float(time_to_reply),
            "reply_model": Selected_Model,
            }
    data_string = json.dumps(dict)
    data = data_string.encode("utf-8")
    future = publisher.publish(reply_pubsub_topic_id, data)
    return (future)
