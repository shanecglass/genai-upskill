# from typing import Literal
from model_mgmt import toolkit
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from model_mgmt import config, prompt
from typing import Literal

# flake8: noqa --E501


tool_node = toolkit.tool_node
model_configs = config.model_to_call()
generative_model = model_configs[0]



def should_continue(state: MessagesState) -> Literal["tools", END]:
    # Define the function that determines whether to continue or not
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"

    # Otherwise, we stop (reply to the user)
    return END

# Define the function that calls the model


def call_model(state: MessagesState):
    input_message = state['messages'][-1].content
    chat_session = config.chat_session
    response = chat_session.invoke(f"{prompt.template} {input_message}")
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def create_graph(state: MessagesState):
    # Define a new graph
    # Define the two nodes we will cycle between
    workflow = StateGraph(state)
    workflow.add_node("chat", call_model)
    workflow.add_node("tools", tool_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.add_edge(START, "chat")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "chat",
        should_continue
        # Next, we pass in the function that will determine which node is called next.
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", 'chat')

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph

    return workflow.compile(checkpointer=checkpointer)
