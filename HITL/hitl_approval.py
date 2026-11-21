from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    decision: str


def ask_human(state: State):

    decision = interrupt("Approve? yes/no")

    return {
        "decision": decision
    }


def route(state: State):

    if state["decision"] == "yes":
        return "approved"

    return "rejected"


def approved(state: State):

    print("Approved")

    return state


def rejected(state: State):

    print("Rejected")

    return state


builder = StateGraph(State)

builder.add_node("human", ask_human)
builder.add_node("approved", approved)
builder.add_node("rejected", rejected)

builder.add_edge(START, "human")

builder.add_conditional_edges(
    "human",
    route,
    {
        "approved": "approved",
        "rejected": "rejected"
    }
)

builder.add_edge("approved", END)
builder.add_edge("rejected", END)

app = builder.compile(
    checkpointer=InMemorySaver()
)


config = {
    "configurable": {
        "thread_id": "1"
    }
}


app.invoke(
    {"decision": ""},
    config
)

user_input = input("Approve? ")

result = app.invoke(
    Command(resume=user_input),
    config
)