from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv



load_dotenv()



# llm defined
llm = ChatGroq(model = "openai/gpt-oss-20b")


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = chroma()