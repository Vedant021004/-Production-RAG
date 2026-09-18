from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader



text = """
LangChain is a framework for building LLM applications.
It provides tools for RAG, agents, chains and more.
"""

loader = TextLoader("data/example.txt")

documents = loader.load()

# splitter = CharacterTextSplitter(chunk_size=200)

# chunks = splitter.split_text(text)

# print("first",chunks)




splitter = CharacterTextSplitter(chunk_size=200)

chunks = splitter.split_documents(documents)
print("second",chunks[0].page_content)