from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
import sqlite3


load_dotenv()


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


# -----------------------------
# SQLite Checkpointer
# -----------------------------

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)


# -----------------------------
# State
# -----------------------------

class State(BaseModel):

    messages: Annotated[list, add_messages]


# -----------------------------
# Chat Node
# -----------------------------

def chat_node(state: State):

    messages = state.messages

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# -----------------------------
# Graph
# -----------------------------

graph = StateGraph(State)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


# -----------------------------
# Compile
# -----------------------------

chatbot = graph.compile(
    checkpointer=checkpointer
)


# -----------------------------
# Choose Thread
# -----------------------------

print("\nAvailable Threads:")
print("1. Thread 1")
print("2. Thread 2")

thread_id = input("\nEnter your thread ID (1 or 2): ")


# Only allow 1 and 2
while thread_id not in ["1", "2"]:

    print("❌ Invalid thread ID. Only 1 or 2 are allowed.")

    thread_id = input("Enter your thread ID (1 or 2): ")


config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# -----------------------------
# Chat Loop
# -----------------------------

while True:

    user = input("\nASK: ")

    if user.lower() in ["exit", "quit"]:

        print("Chat ended.")
        break


    result = chatbot.invoke(
        {
            "messages": [user]
        },
        config=config
    )


    print(
        "AI:",
        result["messages"][-1].content
    )


    