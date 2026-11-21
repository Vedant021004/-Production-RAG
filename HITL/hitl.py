from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    name: str


def ask_name(state: State):

    name = interrupt("Enter your name:")

    return {
        "name": name
    }


builder = StateGraph(State)

builder.add_node("ask_name", ask_name)

builder.add_edge(START, "ask_name")
builder.add_edge("ask_name", END)

app = builder.compile(
    checkpointer=InMemorySaver()
)


config = {
    "configurable": {
        "thread_id": "1"
    }
}


app.invoke(
    {"name": ""},
    config
)

user_input = input("Your name: ")

result = app.invoke(
    Command(resume=user_input),
    config
)

print(result)