from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
load_dotenv()
import os

# Initialsed Variables to that it can be changes further ::
CHUNK_SIZE=500
CHUNK_OVERLAP=150
EMBED_MODEL = "nomic-embed-text:latest"
LLM = "openai/gpt-oss-20b"
QDRNT_URL = os.getenv("QDRNT_URL") ## YOU HAVE TO CHANGE THISSSSSS .. make your acc on qdrant cloud and paste the endpoint id in dotenv file as "Qdrant url"
DIMENSION = "768"
COLLECTION="hybrid_rag_1"
API=os.getenv("QDRANT_API_KEY") # THIS ALSO .. get your own  API of qdrant

## Create llm embedding aur loaders
## PymuPdfLoader is very beneficial as isme image bhi scan hojate h OCR k through
llm=ChatGroq(model=LLM)

embedding=OllamaEmbeddings(model=EMBED_MODEL)

qdr=QdrantClient(
    url=QDRNT_URL,
    api_key=API)

pdf=PyMuPDFLoader(file_path=r"C:\Users\riddh\OneDrive\Desktop\RV\carteye.pdf")
docs=pdf.load()

# CHUNKING
splitter=RecursiveCharacterTextSplitter(
    chunk_size= CHUNK_SIZE,
    chunk_overlap= CHUNK_OVERLAP)

chunks=splitter.split_documents(docs)

# Create collection on qdrant cloud:
if not qdr.collection_exists(COLLECTION):
    qdr.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense" : models.VectorParams(
            size=DIMENSION,
            distance=models.Distance.COSINE
        )},
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )}
    )
## Now store Vectors in db in two format:
### 1. dense  2. sparse
### POINT structure:
#### 1. id- unique id  2.vector-sparse and dense vector   3.payload-metadata

#### text in payload is stored so that it can be given to llm for giving answer and Text in sparse is used for searching purpose .. 
#it will convert text into bm25 sparse matrix and when user asks query it is converted into vector and is given to dense and sparse search for comparison
texts=[]
for chunk in chunks:
    texts.append(chunk.page_content)
# print(texts[1])

vectors=embedding.embed_documents(texts)
print("Dimesion to take: ",len(vectors[0]))
# print(vectors)

points=[]
for i , vect in enumerate(vectors):
    payload={**chunks[i].metadata,## ** used to unpack the doctionaries
             "text": texts[i]}
    point=models.PointStruct(
        id=i,
        vector={
            "dense":vect,##vectors that we created by ollamaembeddings
            "bm25":models.Document(text=texts[i], model="Qdrant/bm25")
        },
        payload=payload
    )
    points.append(point)
# print(len(points))
# print(points[1])
qdr.upsert(collection_name=COLLECTION, points=points)

#Implementing Hybrid search:
#we will pass query and the value of k (how many max doc we want) and we will prefetch them first with dense search and then with sparse search and 
#we will pass it through RRF fusion . In prefetch we can pass the list of queries
#Prefetch basically means bringing results from different search methods.. 
#So for dense search we pass the query vector and for sparse we pass the raw text i.e. raw query
def HybridSearch(query,k=3):
    q_vec= embedding.embed_query(query)
    records= qdr.query_points(
        collection_name= COLLECTION,
        prefetch=[
            models.Prefetch(
                query=q_vec,
                using="dense",
                limit=5     
            ),
            models.Prefetch(
                query=models.Document(text=query,model="Qdrant/bm25"),
                using="bm25",
                limit=5
            )   
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True
    )
    return records.points
# No we will write our query and pass it in hybrid search function then the function will give us some relevant chunks 
# and we will the pass the chunks as CONTEXT and query as QUESTION to our llm model
query="List me the name of the project, team and participants"
relv_chunks=HybridSearch(query)

context=""
for chunk in relv_chunks:
    context+=chunk.payload.get("text")+"/n"

prompt=f""" 
You are an assistant that gives the answer of user questions with provided context based on user documents.
Question: {query},
Context: {context} 
"""
res=llm.invoke(prompt)
print(res.content)
