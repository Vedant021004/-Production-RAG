from typing import Annotated

from pydantic import BaseModel

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


load_dotenv()


# =========================
# LLM
# =========================

model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# =========================
# STATE
# =========================

class State(BaseModel):

    query: str

    messages: Annotated[list, add_messages]


# =========================
# PDF
# =========================

loader = PyPDFLoader("doc.pdf")

docs = loader.load()


# =========================
# CHUNKING
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)


# =========================
# EMBEDDINGS
# =========================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =========================
# CHROMA
# =========================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vedant_db"
)


# =========================
# RETRIEVER
# =========================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# =========================
# RAG TOOL
# =========================

@tool
def search_pdf(query: str):
    """Search the PDF for relevant information."""

    results = retriever.invoke(query)

    return results


# =========================
# RAG NODE
# =========================

def rag_node(state: State):

    results = search_pdf.invoke({
        "query": state.query
    })

    prompt = f"""
    Answer the question using the retrieved PDF information.

    Retrieved information:
    {results}

    Question:
    {state.query}

    If the answer is not present in the retrieved information,
    say that you could not find the answer in the document.
    """

    response = model.invoke(prompt)

    return {
        "messages": [response]
    }


# =========================
# GRAPH
# =========================

graph = StateGraph(State)

graph.add_node("rag", rag_node)

graph.add_edge(START, "rag")

graph.add_edge("rag", END)


# =========================
# COMPILE
# =========================

app = graph.compile()


# =========================
# INPUT
# =========================

question = input("ASK: ")


# =========================
# RUN
# =========================

result = app.invoke(
    State(
        query=question,
        messages=[]
    )
)


# =========================
# OUTPUT
# =========================

print("\nANSWER:")

print(result["messages"][-1].content)