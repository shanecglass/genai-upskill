from dataclasses import dataclass
from langgraph.graph.message import add_messages
from model_calls import ask_gemma, function_coordination
from model_mgmt import config
from operator import add
from typing import Annotated, Literal, TypedDict, Literal
from vertexai.generative_models import ChatSession

import datetime

import logging
import mesop as me
import modules
import time
import uuid
import vertexai

# flake8: noqa --E501


Role = Literal["user", "assistant"]
Selected_Model = config.Selected_Model
model_configs = config.model_to_call(Selected_Model)
generative_model = config.generative_model
endpoint_id = config.endpoint_id
# tool_node = toolkit.tool_node
session_id = str(uuid.uuid4())

@dataclass(kw_only=True)
class ChatMessage:
    """Chat message metadata."""

    role: Role = "user"
    content: str = ""
    edited: bool = False


@me.stateclass
class State:
    input: str
    output: list[ChatMessage]
    in_progress: bool
    message_count: int = 0
    reply_count: int = 0
    session_id: str = session_id

class Chat_State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    # session: str = session_id
    messages: Annotated[list, add_messages]
    chat_session: ChatSession = config.chat_session


def load(e: me.LoadEvent):
    me.set_theme_mode("dark")
    vertexai.init(project=config.project_id, location=config.location)
    global chat_session
    chat_session = config.start_chat(generative_model)
    Chat_State.chat_session = chat_session
    yield

@me.page(
    on_load=load,
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=[
            "https://google.github.io", "https://huggingface.co"],
        dangerously_disable_trusted_types=True
    ),
    path="/",
    title="TravelChat",
)


def page():
    state = me.state(State)
    # Chat UI
    with me.box(style=_STYLE_APP_CONTAINER):
        me.text(_TITLE, type="headline-5", style=_STYLE_TITLE)
        with me.box(style=_STYLE_CHAT_BOX):
            for index, msg in enumerate(state.output):
                with me.box(
                    style=_make_style_chat_bubble_wrapper(msg.role),
                    key=f"msg-{index}",
                ):
                    if msg.role == _ROLE_ASSISTANT:
                        me.text(
                            _display_username(_BOT_USER_DEFAULT, msg.edited),
                            style=_STYLE_CHAT_BUBBLE_NAME,
                        )
                    with me.box(style=_make_chat_bubble_style(msg.role, msg.edited)):
                        if msg.role == _ROLE_USER:
                            me.text(msg.content, style=_STYLE_CHAT_BUBBLE_PLAINTEXT)
                        else:
                            me.markdown(msg.content)
                            with me.tooltip(message="Rewrite response"):
                                me.icon(icon="edit_note")

            if state.in_progress:
                with me.box(key="scroll-to", style=me.Style(height=250)):
                    pass
        with me.box(style=_STYLE_CHAT_INPUT_BOX):
            with me.box(style=me.Style(flex_grow=1)):
                me.input(
                    label=_LABEL_INPUT,
                    # Workaround: update key to clear input.
                    key=f"input-{len(state.output)}",
                    on_input=on_chat_input,
                    on_enter=on_click_submit_chat_msg,
                    style=_STYLE_CHAT_INPUT,
                )
            with me.content_button(
                color="primary",
                type="flat",
                disabled=state.in_progress,
                on_click=on_click_submit_chat_msg,
                style=_STYLE_CHAT_BUTTON,
            ):
                me.icon(
                    _LABEL_BUTTON_IN_PROGRESS if state.in_progress else _LABEL_BUTTON
                )
# Event Handlers

def on_chat_input(e: me.InputEvent):
    """Capture chat text input."""
    state = me.state(State)
    state.input = e.value


def on_click_submit_chat_msg(e: me.ClickEvent | me.InputEnterEvent):
    state = me.state(State)
    if state.in_progress or not state.input:
      return
    input = state.input
    state.input = ""
    yield

    submit_time = time.time()
    submit_time_bq_format = submit_time*pow(10, 6)
    submit_time_human = datetime.datetime.fromtimestamp(submit_time)
    logging.info(f"User message submitted as: {
                 input} at time: {submit_time_human}")
    state.message_count = state.message_count + 1
    modules.publish_message_pubsub(
        input, submit_time_bq_format, state.message_count, state.session_id)
    logging.info(f"PubSub message for user message successfully sent for message {
                 state.message_count} in session {state.session_id}")

    output = state.output
    if output is None:
      output = []
    output.append(ChatMessage(role=_ROLE_USER, content=input))
    state.in_progress = True
    me.scroll_into_view(key="scroll-to")
    yield

    start_time = time.time()
    output_message = respond_to_chat(input, state.output)
    assistant_message = ChatMessage(role=_ROLE_ASSISTANT)
    output.append(assistant_message)
    state.output = output

    state.reply_count = state.reply_count + 1
    reply_time = time.time()
    response_time = reply_time - submit_time
    reply_time_bq_format = submit_time*pow(10, 6)
    reply_time_human = datetime.datetime.fromtimestamp(reply_time)
    logging.info(f"Model PubSub message sent as: {
        output_message} at time: {reply_time_human}")
    logging.info(f"Response time in seconds: {response_time}")
    modules.publish_reply_pubsub(
        output_message, reply_time_bq_format, state.reply_count, state.session_id, response_time)
    logging.info(f"PubSub message for reply successfully sent for message {
        state.reply_count} in session {state.session_id}")


    for content in output_message:
      assistant_message.content += content
      # TODO: 0.25 is an abitrary choice. In the future, consider making this adjustable.
      if (time.time() - start_time) >= 25:
        start_time = time.time()
    yield output_message
    state.in_progress = False
    yield

# Transform function for processing chat messages.

def respond_to_chat(input: str, history: list[ChatMessage]):
    state = me.state(State)
    chat_history = ""
    for h in history:
        chat_history += "<start_of_turn>{role}{content}<end_of_turn>".format(
            role=h.role, content=h.content)
    if Selected_Model == config.Valid_Models["GEMMA"]:
        # Assemble prompt from chat history
        full_input = f"{chat_history}\n{input}"

        result = ask_gemma(full_input)
        yield result
    else:
        # Assemble prompt from chat history if selected model is Gemini Tuned or Gemini
        # if state.reply_count == 0:
        #     full_input = input
        # else:
        #
        # full_input = "\n".join(message.content for message in history)
        # full_input = f"{chat_history}\n{input}"
        chat_session = Chat_State.chat_session
        result = function_coordination(input, chat_session=chat_session)
        yield result

        # result = ask_gemini(full_input)
        # input_message = {"messages": [HumanMessage(content=input)]}
        # input_config = {"configurable": {"thread_id": session_id}}
        # response = app.invoke(input_message, input_config)
        # result = response["messages"][-1].content
        # logging.info(f'Response to {input}: {result}')
        # print(type(result))

        # if type(result) is str:
        #     result = result
        #     return result
        # else:
        #     try:
        #         result = str(result)
        #         return result
        #     except TypeError:
        #         logging.info(f"Type Error occurred on final output to chat. {TypeError}")
        #         return("Uh, that didn't go well. Try again!")


# Constants

_TITLE = "TravelChat"

_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"
_ROLE_SYSTEM = "system"

_BOT_USER_DEFAULT = "travelchat-bot"


# Styles

_COLOR_BACKGROUND = me.theme_var("background")
_COLOR_CHAT_BUBBLE_YOU = me.theme_var("surface-container-low")
_COLOR_CHAT_BUBBLE_BOT = me.theme_var("secondary-container")
_COLOR_CHAT_BUUBBLE_EDITED = me.theme_var("tertiary-container")

_DEFAULT_PADDING = me.Padding.all(20)
_DEFAULT_BORDER_SIDE = me.BorderSide(
    width="1px", style="solid", color=me.theme_var("secondary-fixed")
)

_LABEL_BUTTON = "send"
_LABEL_BUTTON_IN_PROGRESS = "pending"
_LABEL_INPUT = "Where would you like to travel?"

_STYLE_INPUT_WIDTH = me.Style(width="100%")

_STYLE_APP_CONTAINER = me.Style(
    background=_COLOR_BACKGROUND,
    display="flex",
    flex_direction="column",
    height="100%",
    margin=me.Margin.symmetric(vertical=0, horizontal="auto"),
    width="min(1024px, 100%)",
    box_shadow="0 3px 1px -2px #0003, 0 2px 2px #00000024, 0 1px 5px #0000001f",
    padding=me.Padding(top=20, left=20, right=20),
)
_STYLE_TITLE = me.Style(padding=me.Padding(left=10))
_STYLE_CHAT_BOX = me.Style(
    flex_grow=1,
    overflow_y="scroll",
    padding=_DEFAULT_PADDING,
    margin=me.Margin(bottom=20),
    border_radius="10px",
    border=me.Border(
        left=_DEFAULT_BORDER_SIDE,
        right=_DEFAULT_BORDER_SIDE,
        top=_DEFAULT_BORDER_SIDE,
        bottom=_DEFAULT_BORDER_SIDE,
    ),
)
_STYLE_CHAT_INPUT = me.Style(width="100%")
_STYLE_CHAT_INPUT_BOX = me.Style(
    padding=me.Padding(top=30), display="flex", flex_direction="row"
)
_STYLE_CHAT_BUTTON = me.Style(margin=me.Margin(top=8, left=8))
_STYLE_CHAT_BUBBLE_NAME = me.Style(
    font_weight="bold",
    font_size="12px",
    padding=me.Padding(left=15, right=15, bottom=5),
)
_STYLE_CHAT_BUBBLE_PLAINTEXT = me.Style(margin=me.Margin.symmetric(vertical=15))

_STYLE_MODAL_CONTAINER = me.Style(
    background=me.theme_var("surface-container"),
    margin=me.Margin.symmetric(vertical="0", horizontal="auto"),
    width="min(1024px, 100%)",
    box_sizing="content-box",
    height="100%",
    overflow_y="scroll",
    box_shadow=("0 3px 1px -2px #0003, 0 2px 2px #00000024, 0 1px 5px #0000001f"),
)

_STYLE_MODAL_CONTENT = me.Style(margin=me.Margin.all(20))

_STYLE_PREVIEW_CONTAINER = me.Style(
    display="grid",
    grid_template_columns="repeat(2, 1fr)",
)

_STYLE_PREVIEW_ORIGINAL = me.Style(
    color=me.theme_var("on-surface"), padding=_DEFAULT_PADDING
)

_STYLE_PREVIEW_REWRITE = me.Style(
    background=_COLOR_CHAT_BUUBBLE_EDITED, padding=_DEFAULT_PADDING
)


def _make_style_chat_bubble_wrapper(role: Role) -> me.Style:
    """Generates styles for chat bubble position.

    Args:
        role: Chat bubble alignment depends on the role
    """
    align_items = "end" if role == _ROLE_USER else "start"
    return me.Style(
        display="flex",
        flex_direction="column",
        align_items=align_items,
    )


def _make_chat_bubble_style(role: Role, edited: bool) -> me.Style:
    """Generates styles for chat bubble.

    Args:
        role: Chat bubble background color depends on the role
        edited: Whether chat message was edited or not.
    """
    background = _COLOR_CHAT_BUBBLE_YOU
    if role == _ROLE_ASSISTANT:
        background = _COLOR_CHAT_BUBBLE_BOT
    if edited:
        background = _COLOR_CHAT_BUUBBLE_EDITED

    return me.Style(
        width="80%",
        font_size="13px",
        background=background,
        border_radius="15px",
        padding=me.Padding(right=15, left=15, bottom=3),
        margin=me.Margin(bottom=10),
        border=me.Border(
            left=_DEFAULT_BORDER_SIDE,
            right=_DEFAULT_BORDER_SIDE,
            top=_DEFAULT_BORDER_SIDE,
            bottom=_DEFAULT_BORDER_SIDE,
        ),
    )

def _display_username(username: str, edited: bool = False) -> str:
    """Displays the username

    Args:
        username: Name of the user
        edited: Whether the message has been edited.
    """
    edited_text = " (edited)" if edited else ""
    return username + edited_text
