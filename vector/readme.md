# 🔎 Vector Search, Similarity Search & Cosine Similarity

A beginner-friendly guide to understanding **Vector Search**, **Similarity Search**, **Cosine Similarity**, and their implementation using **LangChain + Chroma**.

---

## 📚 Table of Contents

* [What is Vector Search?](#-what-is-vector-search)
* [Why Vector Search?](#-why-vector-search)
* [Embeddings](#-embeddings)
* [Vectors](#-vectors)
* [Similarity Search](#-similarity-search)
* [Cosine Similarity](#-cosine-similarity)
* [How Vector Search Works](#-how-vector-search-works)
* [Vector Store](#-vector-store)
* [Chroma](#-chroma)
* [Important Vector Store Functions](#-important-vector-store-functions)
* [Similarity Search with LangChain](#-similarity-search-with-langchain)
* [Similarity Search with Score](#-similarity-search-with-score)
* [Search by Vector](#-search-by-vector)
* [Retriever](#-retriever)
* [MMR Search](#-mmr-search)
* [Vector Search in RAG](#-vector-search-in-rag)
* [Complete Example](#-complete-example)
* [Cheat Sheet](#-cheat-sheet)

---

# 🧠 What is Vector Search?

**Vector Search** is a search technique that finds information based on the **meaning of the query**, rather than only matching exact keywords.

For example:

### Query

```text
How can I learn Python?
```

### Document

```text
Python is widely used for software development,
automation and data science.
```

The exact words aren't identical, but the **meaning is related**.

Vector search can identify this relationship by converting text into numerical vectors called **embeddings**.

```text
Text
 ↓
Embedding Model
 ↓
Vector
 ↓
Vector Database
 ↓
Similarity Search
 ↓
Relevant Documents
```

---

# 🔥 Why Vector Search?

Traditional keyword search might work like:

```text
"How can I learn Python?"
             ↓
       Find "Python"
```

Vector search works more like:

```text
"How can I learn Python?"
             ↓
      Understand meaning
             ↓
          Vector
             ↓
       Find similar vectors
```

This is especially useful for:

* RAG
* Semantic Search
* Question Answering
* Recommendation Systems
* Document Search
* AI Assistants
* Knowledge Bases

---

# 🧩 Embeddings

An **embedding** converts text into a numerical representation.

```text
"Python is easy"
        ↓
Embedding Model
        ↓
[0.21, 0.72, -0.14, 0.63, ...]
```

The resulting vector can have hundreds or thousands of dimensions depending on the embedding model.

Example:

```python
embedding = embeddings.embed_query(
    "What is Python?"
)

print(embedding)
```

---

# 📐 Vectors

A vector is simply a collection of numbers.

Example:

```python
A = [1, 2, 3]
B = [2, 3, 4]
```

For embeddings:

```text
Document
   ↓
Embedding
   ↓
[0.12, 0.83, 0.21, 0.64, ...]
```

We don't normally interpret each dimension individually.

Instead, we compare vectors to determine how similar their representations are.

---

# 🔍 Similarity Search

**Similarity Search** finds the vectors that are most similar to a query vector.

Suppose we have:

```text
Document A → Vector A
Document B → Vector B
Document C → Vector C
Document D → Vector D
```

User asks:

```text
"What is Python?"
```

The query becomes:

```text
Query
 ↓
Embedding
 ↓
Query Vector
```

Then the vector is compared against the stored vectors.

```text
Document A → 0.91
Document B → 0.82
Document C → 0.63
Document D → 0.41
```

If we request:

```python
k = 2
```

the system returns the top 2 results.

```text
Document A
Document B
```

---

# 📐 Cosine Similarity

One common similarity measure is **Cosine Similarity**.

It measures the **angle between two vectors**.

Conceptually:

```text
Similar direction
       ↓
High similarity
```

```text
Different direction
       ↓
Low similarity
```

The formula is:

$$
\text{Cosine Similarity}(A,B)
=
\frac{A \cdot B}
{\|A\|\|B\|}
$$

Where:

* `A · B` = dot product
* `||A||` = magnitude of vector A
* `||B||` = magnitude of vector B

For two vectors:

```text
A = [a1, a2, a3]

B = [b1, b2, b3]
```

the dot product is:

```text
A · B = a1*b1 + a2*b2 + a3*b3
```

### Typical interpretation

```text
+1 → same direction
 0 → perpendicular
-1 → opposite direction
```

For many embedding systems, vectors are often normalized, which makes cosine similarity closely related to the dot product.

---

# ⚙️ How Vector Search Works

The complete process:

```text
                 USER QUERY
                      │
                      ▼
              "What is Python?"
                      │
                      ▼
              EMBEDDING MODEL
                      │
                      ▼
                QUERY VECTOR
                      │
                      ▼
              ┌───────────────┐
              │  VECTOR STORE │
              └───────────────┘
                      │
                      ▼
              SIMILARITY SEARCH
                      │
                      ▼
               RANK RESULTS
                      │
                      ▼
                TOP K DOCS
```

---

# 🗄️ Vector Store

A **Vector Store** stores embeddings and provides methods for searching them.

Popular vector databases/stores include:

* Chroma
* FAISS
* Pinecone
* Qdrant
* Weaviate
* Milvus

A stored record commonly contains:

```text
Document
   +
Embedding
   +
Metadata
```

Example:

```python
Document(
    page_content="Python is a programming language.",
    metadata={
        "source": "python.txt"
    }
)
```

---

# 🟣 Chroma

For this project we can use **Chroma**.

Install:

```bash
pip install langchain-chroma chromadb
```

Import:

```python
from langchain_chroma import Chroma
```

Create a vector store:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

### Parameters

| Parameter            | Purpose                       |
| -------------------- | ----------------------------- |
| `collection_name`    | Name of the vector collection |
| `embedding_function` | Converts text into vectors    |
| `persist_directory`  | Stores Chroma data locally    |

---

# 🛠️ Important Vector Store Functions

Some important LangChain VectorStore methods are:

| Function                                    | Purpose                            |
| ------------------------------------------- | ---------------------------------- |
| `add_documents()`                           | Add LangChain Documents            |
| `add_texts()`                               | Add raw text                       |
| `delete()`                                  | Delete stored data                 |
| `get_by_ids()`                              | Retrieve documents by ID           |
| `similarity_search()`                       | Search similar documents           |
| `similarity_search_with_score()`            | Search with scores                 |
| `similarity_search_with_relevance_scores()` | Search with relevance scores       |
| `similarity_search_by_vector()`             | Search using an existing vector    |
| `as_retriever()`                            | Convert VectorStore into Retriever |

---

# 🔎 Similarity Search with LangChain

Basic similarity search:

```python
docs = vector_store.similarity_search(
    "What is Python?",
    k=3
)
```

### `k`

`k` determines how many documents should be returned.

```python
k=1
```

returns one document.

```python
k=5
```

returns five documents.

Example:

```python
for doc in docs:
    print(doc.page_content)
```

---

# 📊 Similarity Search with Score

You can also retrieve scores:

```python
results = vector_store.similarity_search_with_score(
    "What is Python?",
    k=3
)
```

Then:

```python
for doc, score in results:
    print("Score:", score)
    print("Content:", doc.page_content)
    print()
```

⚠️ **Important:** Don't assume that every vector store uses the same score scale or that a larger score always means greater similarity. The exact score semantics depend on the vector store and distance metric.

---

# 🧮 Search by Vector

You can manually create the query embedding first:

```python
query_vector = embeddings.embed_query(
    "What is Python?"
)
```

Then search:

```python
docs = vector_store.similarity_search_by_vector(
    query_vector,
    k=3
)
```

Flow:

```text
Text
 ↓
Embedding Model
 ↓
Query Vector
 ↓
Vector Search
 ↓
Documents
```

---

# 🔄 Retriever

A Retriever provides an interface for retrieving relevant documents.

Convert a VectorStore into a Retriever:

```python
retriever = vector_store.as_retriever()
```

Then:

```python
docs = retriever.invoke(
    "What is Python?"
)
```

You can control the number of results:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)
```

Then:

```python
docs = retriever.invoke(
    "What is Python?"
)
```

---

# 🆚 Vector Store vs Retriever

### Vector Store

Handles:

```text
Store
Search
Delete
Retrieve
```

Example:

```python
vector_store.similarity_search(
    query,
    k=3
)
```

### Retriever

Provides a simpler retrieval interface:

```python
retriever.invoke(query)
```

Think of it as:

```text
VECTOR STORE
     │
     │ as_retriever()
     ▼
  RETRIEVER
     │
     │ invoke()
     ▼
RELEVANT DOCUMENTS
```

---

# 🧠 MMR Search

MMR stands for:

> **Maximum Marginal Relevance**

Normal similarity search mainly focuses on relevance.

MMR tries to balance:

```text
Relevance
    +
Diversity
```

Example:

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3
    }
)
```

This can be useful when your retrieved documents contain a lot of duplicated or highly similar information.

---

# 📄 Adding Documents

Create documents:

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Python is a programming language.",
        metadata={"source": "python.txt"}
    ),

    Document(
        page_content="Chroma is a vector database.",
        metadata={"source": "chroma.txt"}
    ),

    Document(
        page_content="RAG retrieves relevant documents before generating an answer.",
        metadata={"source": "rag.txt"}
    )
]
```

Add them:

```python
vector_store.add_documents(documents)
```

The process:

```text
Documents
    ↓
Embedding Model
    ↓
Embeddings
    ↓
Vector Store
```

---

# 🧪 Complete Example

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# ---------------------------
# 1. Create documents
# ---------------------------

documents = [
    Document(
        page_content="Python is a programming language.",
        metadata={"source": "python.txt"}
    ),

    Document(
        page_content="Chroma is a vector database.",
        metadata={"source": "chroma.txt"}
    ),

    Document(
        page_content="RAG retrieves relevant documents before generating an answer.",
        metadata={"source": "rag.txt"}
    )
]


# ---------------------------
# 2. Create embeddings
# ---------------------------

embeddings = OpenAIEmbeddings()


# ---------------------------
# 3. Create vector store
# ---------------------------

vector_store = Chroma(
    collection_name="learning",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# ---------------------------
# 4. Add documents
# ---------------------------

vector_store.add_documents(documents)


# ---------------------------
# 5. Similarity search
# ---------------------------

query = "What is Python?"

results = vector_store.similarity_search(
    query,
    k=2
)


# ---------------------------
# 6. Display results
# ---------------------------

for doc in results:

    print("CONTENT:")
    print(doc.page_content)

    print("\nMETADATA:")
    print(doc.metadata)

    print("----------------------")
```

---

# 🤖 Vector Search in RAG

Vector search is one of the core components of RAG.

```text
                  USER
                   │
                   ▼
                 QUERY
                   │
                   ▼
             EMBEDDING MODEL
                   │
                   ▼
              QUERY VECTOR
                   │
                   ▼
             VECTOR STORE
                   │
                   ▼
            SIMILARITY SEARCH
                   │
                   ▼
           RELEVANT DOCUMENTS
                   │
                   ▼
                 PROMPT
                   │
                   ▼
                  LLM
                   │
                   ▼
                ANSWER
```

So:

```text
RAG
=
Retrieval
+
Context
+
Generation
```

And vector search is commonly used for the **retrieval** part.

---

# ⚡ Vector Search Cheat Sheet

### Convert text to vector

```python
embedding = embeddings.embed_query(
    "What is Python?"
)
```

### Add Documents

```python
vector_store.add_documents(documents)
```

### Add text

```python
vector_store.add_texts(texts)
```

### Similarity Search

```python
vector_store.similarity_search(
    query,
    k=3
)
```

### Similarity Search + Score

```python
vector_store.similarity_search_with_score(
    query,
    k=3
)
```

### Search using an existing vector

```python
vector_store.similarity_search_by_vector(
    query_vector,
    k=3
)
```

### Create Retriever

```python
retriever = vector_store.as_retriever()
```

### Retrieve

```python
docs = retriever.invoke(query)
```

### MMR

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)
```

---

# 🧠 Final Mental Model

Remember this:

```text
                    TEXT
                     │
                     ▼
               EMBEDDING MODEL
                     │
                     ▼
                   VECTOR
                     │
                     ▼
              ┌──────────────┐
              │ VECTOR STORE │
              └──────────────┘
                     │
                     ▼
             SIMILARITY SEARCH
                     │
                     ▼
              TOP K DOCUMENTS
                     │
                     ▼
                 RETRIEVER
                     │
                     ▼
                    LLM
                     │
                     ▼
                  ANSWER
```

### The key idea:

> **Text → Embedding → Vector → Similarity Search → Relevant Documents → LLM**

This is the foundation you need before moving deeper into **RAG, hybrid search, reranking, metadata filtering, and advanced retrieval**. 🚀
