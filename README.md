# Production RAG

> A production-oriented journey through **Retrieval-Augmented Generation (RAG)** — from document ingestion and chunking to advanced retrieval, multimodal RAG, agentic workflows, evaluation, and deployment.

This repository is being built step-by-step to understand **how modern RAG systems actually work internally**, rather than treating RAG as a simple `load → embed → retrieve → generate` pipeline.

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with Large Language Models.

Instead of asking an LLM to answer only from its internal knowledge:

```text
User Query
    ↓
LLM
    ↓
Answer
```

RAG first retrieves relevant information from an external knowledge base:

```text
                    ┌──────────────┐
                    │   Documents  │
                    └──────┬───────┘
                           ↓
                    Document Loader
                           ↓
                        Chunking
                           ↓
                       Embeddings
                           ↓
                     Vector Database
                           ↓
User Query → Retrieval → Relevant Context
                           ↓
                          LLM
                           ↓
                        Answer
```

The goal of this repository is to progressively build this system into a **reliable, observable and production-ready RAG platform**.

---

# 🎯 Repository Goals

This project covers:

* Document ingestion
* Document parsing
* Text cleaning
* Chunking
* Metadata management
* Embeddings
* Vector databases
* Similarity search
* MMR retrieval
* Metadata filtering
* Hybrid search
* Query transformation
* Multi-query retrieval
* RAG Fusion
* HyDE
* Reranking
* Context compression
* Parent-child retrieval
* Multi-document RAG
* Multimodal RAG
* Corrective RAG
* Self-RAG
* Agentic RAG
* RAG evaluation
* Observability
* Production deployment

---

# 🗺️ RAG Roadmap

```text
                    RAG ENGINEERING ROADMAP

                           RAG
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
   INGESTION            RETRIEVAL           GENERATION
        │                   │                   │
   Loaders              Similarity           Prompting
   Parsing              MMR                  Citations
   Cleaning             Metadata             Structured Output
   Chunking             Hybrid Search
   Metadata             Query Rewrite
                        Multi Query
                        HyDE
                        RAG Fusion
                        Reranking
                             │
                             ↓
                       ADVANCED RAG
                             │
                ┌────────────┼────────────┐
                ↓            ↓            ↓
             CRAG         Self-RAG     Agentic RAG
                │            │            │
                └────────────┼────────────┘
                             ↓
                      MULTIMODAL RAG
                             │
                  ┌──────────┼──────────┐
                  ↓          ↓          ↓
                 Text       PDF       Images
                  ↓          ↓          ↓
                Tables      Audio      Video
                             │
                             ↓
                         EVALUATION
                             │
                ┌────────────┼────────────┐
                ↓            ↓            ↓
             Retrieval    Generation    System
              Metrics      Metrics      Metrics
                │            │            │
                └────────────┼────────────┘
                             ↓
                       PRODUCTION RAG
                             │
                  API + Docker + Cloud
```

---

# 📂 Repository Structure

The repository will evolve as the RAG system grows.

```text
rag/
│
├── 01_document_loading/
│   ├── pdf/
│   ├── csv/
│   ├── txt/
│   ├── markdown/
│   ├── json/
│   ├── html/
│   ├── word/
│   └── github/
│
├── 02_document_parsing/
│
├── 03_cleaning/
│
├── 04_chunking/
│
├── 05_embeddings/
│
├── 06_vector_databases/
│   ├── chroma/
│   ├── qdrant/
│   └── faiss/
│
├── 07_retrieval/
│   ├── similarity_search/
│   ├── mmr/
│   ├── metadata_filtering/
│   └── hybrid_search/
│
├── 08_query_transformation/
│   ├── query_rewriting/
│   ├── multi_query/
│   ├── hyde/
│   └── rag_fusion/
│
├── 09_reranking/
│
├── 10_context_compression/
│
├── 11_advanced_rag/
│   ├── parent_child/
│   ├── multi_document/
│   ├── crag/
│   ├── self_rag/
│   └── agentic_rag/
│
├── 12_multimodal_rag/
│   ├── text/
│   ├── images/
│   ├── pdf/
│   ├── tables/
│   ├── audio/
│   └── video/
│
├── 13_rag_agents/
│   ├── langchain/
│   └── langgraph/
│
├── 14_evaluation/
│   ├── retrieval/
│   ├── generation/
│   └── end_to_end/
│
├── 15_observability/
│
├── 16_production/
│   ├── fastapi/
│   ├── docker/
│   ├── caching/
│   ├── authentication/
│   └── deployment/
│
├── projects/
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# 01 — 📥 Document Loading

The first stage of RAG is getting information into the system.

```text
External Data
     ↓
Document Loader
     ↓
LangChain Document
     ↓
Parser
     ↓
Cleaner
```

A typical LangChain document contains:

```python
Document(
    page_content="...",
    metadata={
        "source": "...",
        "page": 1
    }
)
```

---

## 📚 Document Loaders

### PDF

Learn:

* `PyPDFLoader`
* PDF page extraction
* Page metadata
* Scanned PDFs
* Layout-aware parsing
* Tables
* Images

---

### TXT

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("example.txt")

documents = loader.load()
```

---

### CSV

Learn:

* Row-based loading
* Column metadata
* Converting rows into semantic documents
* CSV → RAG pipeline

---

### Markdown

Learn:

* Markdown structure
* Headers
* Sections
* Metadata
* Semantic splitting

---

### JSON

Learn:

* Nested JSON
* Structured data
* Metadata extraction
* JSON → Documents

---

### HTML / Web

Learn:

* Web page loading
* HTML parsing
* Removing unnecessary content
* Metadata extraction

---

### Word Documents

Learn:

* `.docx`
* Paragraph extraction
* Tables
* Metadata

---

### GitHub

Git repositories can also become knowledge sources.

```text
GitHub Repository
       ↓
Git Loader
       ↓
Source Files
       ↓
Documents
       ↓
Chunking
       ↓
Embeddings
       ↓
Vector DB
```

This enables applications such as:

> "Explain the authentication system in this repository."

---

# 🧩 Document Loader Concepts

Before moving forward, understand:

### 1. `load()`

Loads all documents.

```python
documents = loader.load()
```

### 2. `lazy_load()`

Loads documents incrementally.

Useful for large datasets.

### 3. Metadata

Metadata tells the RAG system where the content came from.

Example:

```python
{
    "source": "company_policy.pdf",
    "page": 12,
    "department": "HR"
}
```

Metadata becomes extremely important later for:

* Filtering
* Citations
* Access control
* Debugging
* Retrieval
* Observability

---

# 02 — 🧹 Document Processing

Raw documents are rarely ready for retrieval.

```text
Raw Document
     ↓
Parsing
     ↓
Cleaning
     ↓
Normalization
     ↓
Metadata
     ↓
Processed Document
```

Topics:

* Removing unnecessary text
* Whitespace normalization
* Headers/footers
* Duplicate content
* Encoding
* Metadata normalization
* Document validation

---

# 03 — ✂️ Chunking

Large documents need to be divided into smaller pieces.

```text
Document
   ↓
Chunker
   ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
```

Study:

* Character splitting
* Recursive splitting
* Token-based splitting
* Sentence splitting
* Semantic chunking
* Markdown splitting
* HTML splitting
* Parent-child chunking

Important parameters:

```text
chunk_size
chunk_overlap
separator
```

---

# 04 — 🧮 Embeddings

Convert text into vectors.

```text
"How does authentication work?"
              ↓
         Embedding Model
              ↓
     [0.12, -0.43, 0.91, ...]
```

Study:

* Sentence Transformers
* OpenAI embeddings
* Hugging Face embeddings
* Embedding dimensions
* Cosine similarity
* Distance metrics
* Embedding normalization

---

# 05 — 🗄️ Vector Databases

Store and retrieve embeddings.

Study:

* Chroma
* FAISS
* Qdrant
* Pinecone
* Weaviate

Core operations:

```text
ADD
SEARCH
FILTER
UPDATE
DELETE
```

---

# 06 — 🔎 Retrieval

Retrieval is the heart of RAG.

Basic:

```text
Query
 ↓
Embedding
 ↓
Vector Search
 ↓
Top-K Documents
```

Study:

### Similarity Search

Retrieve semantically similar documents.

### MMR

Maximum Marginal Relevance.

Balances:

```text
Relevance + Diversity
```

### Metadata Filtering

Example:

```text
department = "engineering"
```

### Hybrid Search

Combines:

```text
Dense Retrieval
       +
Sparse Retrieval
       ↓
Hybrid Results
```

---

# 07 — 🔄 Query Transformation

The user's query isn't always the best retrieval query.

Study:

```text
Query Rewriting
Multi Query
HyDE
RAG Fusion
Step-back prompting
Query decomposition
```

Example:

```text
User:
"Why did the model fail?"

        ↓

Generated queries:

"What caused model failure?"
"What errors occurred during inference?"
"What were the model's limitations?"
```

---

# 08 — 🏆 Reranking

Initial retrieval may return:

```text
20 documents
```

A reranker can reorder them:

```text
Retriever
   ↓
20 documents
   ↓
Reranker
   ↓
Top 5 relevant documents
```

Study:

* Cross-encoder rerankers
* Cohere Rerank
* BGE rerankers
* Relevance scoring

---

# 09 — 🧠 Context Compression

Instead of sending entire retrieved documents to the LLM:

```text
Retrieved Documents
        ↓
Relevant Information
        ↓
LLM
```

Benefits:

* Lower token usage
* Lower latency
* Better context quality
* Lower cost

---

# 10 — 🚀 Advanced RAG

This repository will implement:

### CRAG

Corrective Retrieval-Augmented Generation.

```text
Retrieve
   ↓
Evaluate Retrieval
   ↓
Good? ── Yes ──→ Generate
  │
  No
  ↓
Correct / Search Again
```

### Self-RAG

The model evaluates its own retrieval and generation process.

### Agentic RAG

The LLM decides:

```text
Should I retrieve?
Should I search again?
Which source should I use?
Do I need another tool?
Is the answer supported?
```

---

# 11 — 👁️ Multimodal RAG

RAG shouldn't be limited to text.

```text
                 Multimodal RAG
                       │
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
      Text            Image           PDF
       ↓               ↓               ↓
     Tables           Charts          Layout
       │               │               │
       └───────────────┼───────────────┘
                       ↓
                  Unified Context
                       ↓
                      LLM
```

Planned support:

* Text
* PDF
* Images
* Tables
* Charts
* Audio
* Video
* Scanned documents

---

# 12 — 🤖 RAG + LangChain

LangChain will be used for:

* Document loaders
* Document processing
* Retrievers
* Vector stores
* LLM integration
* Chains
* Structured output
* Tool calling

---

# 13 — 🕸️ RAG + LangGraph

LangGraph will be used when RAG becomes a **stateful workflow**.

Example:

```text
             START
               ↓
          Analyze Query
               ↓
        ┌──────┴──────┐
        ↓             ↓
      RAG          Direct LLM
        ↓             ↓
   Evaluate      Generate
        ↓
    Good enough?
      /     \
    Yes      No
     ↓        ↓
 Generate   Retrieve Again
     ↓        │
     └────────┘
          ↓
         END
```

This is where RAG evolves from a pipeline into an **agentic system**.

---

# 14 — 📊 RAG Evaluation

A RAG system isn't good just because it produces fluent answers.

We need to evaluate:

### Retrieval

* Recall
* Precision
* Hit Rate
* MRR
* NDCG

### Generation

* Faithfulness
* Answer relevance
* Context relevance
* Groundedness

### System

* Latency
* Cost
* Token usage
* Retrieval time
* Generation time

Tools:

* RAGAS
* LangSmith
* Custom evaluation datasets

---

# 15 — 🔭 Observability

Production RAG needs visibility into every step.

```text
User Query
    ↓
Query Transformation
    ↓
Retriever
    ↓
Reranker
    ↓
Context
    ↓
LLM
    ↓
Answer
```

Track:

```text
Latency
Tokens
Cost
Retrieved Documents
Scores
Errors
LLM Responses
```

---

# 16 — 🏭 Production RAG

Final architecture:

```text
                       USER
                         │
                         ↓
                    FastAPI
                         │
                         ↓
                 Query Processing
                         │
                         ↓
              Query Transformation
                         │
                         ↓
                    Retriever
                         │
                  ┌──────┴──────┐
                  ↓             ↓
               Vector DB    Keyword Search
                  │             │
                  └──────┬──────┘
                         ↓
                     Reranker
                         ↓
                 Context Compression
                         ↓
                    LLM / Agent
                         ↓
                  Answer + Citations
                         │
                         ↓
                      USER
```

Production technologies:

* FastAPI
* PostgreSQL
* Redis
* Chroma/Qdrant
* Docker
* GitHub Actions
* AWS
* Logging
* Monitoring
* LangSmith
* Authentication
* Rate limiting
* Caching

---

# 🧪 Projects

The repository will eventually contain complete RAG systems.

### Project 01 — Basic RAG

```text
PDF
 ↓
Loader
 ↓
Chunking
 ↓
Embeddings
 ↓
Chroma
 ↓
Retriever
 ↓
LLM
```

### Project 02 — Production RAG

```text
Multi-format Documents
        ↓
Parsing
        ↓
Smart Chunking
        ↓
Embeddings
        ↓
Hybrid Retrieval
        ↓
Reranking
        ↓
Context Compression
        ↓
LLM
        ↓
Citations
        ↓
Evaluation
        ↓
FastAPI
        ↓
Docker
```

### Project 03 — Agentic RAG

```text
                 Agent
                   ↓
          ┌────────┼────────┐
          ↓        ↓        ↓
        Search    RAG     Tools
          ↓        ↓        ↓
          └────────┼────────┘
                   ↓
               Evaluation
                   ↓
                 Answer
```

### Project 04 — Multimodal RAG

```text
PDF + Images + Tables + Text
              ↓
       Multimodal Parser
              ↓
          Processing
              ↓
          Embeddings
              ↓
         Vector Store
              ↓
           Retrieval
              ↓
       Vision + Text LLM
```

---

# 🛠️ Tech Stack

| Category   | Technologies                        |
| ---------- | ----------------------------------- |
| Language   | Python                              |
| Framework  | LangChain                           |
| Workflows  | LangGraph                           |
| LLMs       | OpenAI, Groq, Ollama                |
| Embeddings | Sentence Transformers, Hugging Face |
| Vector DB  | Chroma, Qdrant, FAISS               |
| Reranking  | BGE / Cross-Encoder                 |
| API        | FastAPI                             |
| Database   | PostgreSQL                          |
| Cache      | Redis                               |
| Evaluation | RAGAS, LangSmith                    |
| Deployment | Docker, AWS                         |
| CI/CD      | GitHub Actions                      |

---

# 📚 Learning Philosophy

This repository follows one principle:

> **Don't just use RAG. Understand RAG.**

For every component, understand:

```text
What is it?
     ↓
Why do we need it?
     ↓
How does it work?
     ↓
How do we implement it?
     ↓
What can go wrong?
     ↓
How do we evaluate it?
     ↓
How do we productionize it?
```

---

# 🏁 Current Progress

## Phase 1 — Document Loading

* [x] Understand `Document`
* [x] Understand `page_content`
* [x] Understand metadata
* [ ] TXT Loader
* [ ] PDF Loader
* [ ] CSV Loader
* [ ] JSON Loader
* [ ] Markdown Loader
* [ ] DOCX Loader
* [ ] HTML Loader
* [ ] GitHub Loader

## Phase 2 — Processing

* [ ] Parsing
* [ ] Cleaning
* [ ] Metadata
* [ ] Deduplication

## Phase 3 — Chunking

* [ ] Character splitting
* [ ] Recursive splitting
* [ ] Token splitting
* [ ] Semantic chunking
* [ ] Parent-child chunking

## Phase 4 — Retrieval

* [ ] Embeddings
* [ ] Vector DB
* [ ] Similarity search
* [ ] MMR
* [ ] Metadata filtering
* [ ] Hybrid search
* [ ] Query rewriting
* [ ] Multi-query
* [ ] HyDE
* [ ] RAG Fusion
* [ ] Reranking

## Phase 5 — Advanced RAG

* [ ] CRAG
* [ ] Self-RAG
* [ ] Agentic RAG
* [ ] Multimodal RAG

## Phase 6 — Production

* [ ] Evaluation
* [ ] Observability
* [ ] FastAPI
* [ ] Docker
* [ ] PostgreSQL
* [ ] Redis
* [ ] AWS
* [ ] CI/CD

---

# ⭐ Final Goal

Build a RAG system that can answer:

> **"Where did this information come from, why did you retrieve it, how confident are you, and what happens if the retrieval is wrong?"**

The end goal is not simply a chatbot.

It is a complete:

```text
        ┌──────────────────────────────┐
        │       PRODUCTION RAG         │
        ├──────────────────────────────┤
        │                              │
        │ Ingestion                    │
        │ Parsing                      │
        │ Chunking                     │
        │ Embeddings                   │
        │ Retrieval                    │
        │ Reranking                    │
        │ Query Transformation        │
        │ Multimodal Retrieval         │
        │ Agentic Reasoning            │
        │ Evaluation                   │
        │ Observability                │
        │ APIs                         │
        │ Deployment                   │
        │                              │
        └──────────────────────────────┘
```

**From raw documents → intelligent retrieval → reliable answers → production AI system.**

---

## 👨‍💻 Author

**Vedant Kapil**

AI / ML Engineer · Generative AI · Agentic Systems

---

## ⭐ If this repository helps you

Star the repository and follow the journey as the system evolves from **basic document loading to production-grade RAG**.
