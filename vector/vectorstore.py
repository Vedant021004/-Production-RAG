from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv



load_dotenv()

data = PyPDFLoader("doc.pdf")
docs = data.load()


# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=200,
#     chunk_overlap=50
# )

# # splitter ko store krna hai chunks ke andr
# chunks = splitter.split_documents(docs) 
#  # aur hum doc ko split krenge 

# embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)



# llm defined
llm = ChatGroq(model = "openai/gpt-oss-20b")

vector_store = Chroma.from_documents(
    documents = docs,
    embedding = embeddings
)

retrivers = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" :3, "lambda_mult":1
    }
)

query = "who is vedant kapil"

result = retrivers.invoke(query)

for i, doc in enumerate(result, 1):
    print(f"RESULT {i}")
    print(doc.page_content)