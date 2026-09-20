from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b"
)


class State(BaseModel):
    query: str
    messages: Annotated[list, add_messages]


data = PyPDFLoader("doc.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap =50
)

chunk = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(

    documents=chunk,
    embedding=embeddings,
    persist_directory="./vedant_db"

    )

question = input("ASK: ")

results = vectorstore.similarity_search(
    question,
    k=3

    )

