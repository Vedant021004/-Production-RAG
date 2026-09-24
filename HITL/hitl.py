import os

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)


class State(BaseModel):
    question: str
    answer: str


def llm_node(state: State):

    response = llm.invoke(state.question)

    return {
        "answer": response.content
    }


def human_review(state: State):

    decision = interrupt({
        "question": state.question,
        "answer": state.answer,
        "message": "Approve this answer?"
    })

    if decision == "yes":
        return {}

    return {
        "answer": "Answer rejected by human."
    }


builder = StateGraph(State)

builder.add_node("llm", llm_node)
builder.add_node("human", human_review)

builder.add_edge(START, "llm")
builder.add_edge("llm", "human")
builder.add_edge("human", END)

app = builder.compile(
    checkpointer=InMemorySaver()
)


config = {
    "configurable": {
        "thread_id": "1"
    }
}


result = app.invoke(
    {
        "question": "Explain RAG in simple words.",
        "answer": ""
    },
    config
)

print("Graph paused for human approval.")

result = app.invoke(
    Command(resume="yes"),
    config
)

print(result)