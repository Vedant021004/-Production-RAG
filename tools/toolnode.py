from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


class State(BaseModel):
    messages: Annotated[list, add_messages]


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tools = [add]

tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)


def chat(state: State):

    messages = state.messages

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


graph = StateGraph(State)

graph.add_node("chat", chat)
graph.add_node("bind", tool_node)

graph.add_edge(START, "chat")

graph.add_conditional_edges(
    "chat",
    tools_condition,
    {
        "tools": "bind",
        END: END
    }
)

graph.add_edge("bind", "chat")

result = graph.compile()

user = input("ASK: ")

answer = result.invoke({
    "messages": [
        HumanMessage(content=user)
    ]
})

print("AI:", answer["messages"][-1].content)








from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from typing import TypedDict

class State(TypedDict):
    question: str
    answer: str

def ask_human(state: State):

    decision = interrupt(
        "Do you want to continue?"
    )

    return {
        "answer": decision
    }

graph = StateGraph(State)

graph.add_node("human", ask_human)

graph.add_edge(START, "human")
graph.add_edge("human", END)

app = graph.compile()

config = {
    "configurable": {
        "thread_id": "1"
    }
}

result = app.invoke(
    {
        "question": "Should I continue?",
        "answer": ""
    },
    config
)

print(result)