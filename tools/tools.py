from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
import sqlite3


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

class State(BaseModel):
    question : str
    answer :str



@tool
def tool_one(state:State):

    pass

    
    return {
        "answer" : "haan bhai ho gya"
    }


@tool
def tool_two(state:State):

    question = state.question

    model = llm.invoke(
        f"""answer this {question}"""

    )


    return {
        "answer" : model.content
    }

tools = [tool_one,tool_two]

llm_with_tools = llm.bind_tools(tools)

