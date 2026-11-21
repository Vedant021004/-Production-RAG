# RAG Learning Roadmap: Fundamentals to Production-Ready Systems

A structured, coding-focused roadmap for mastering Retrieval-Augmented Generation (RAG), progressing from foundational concepts to advanced retrieval techniques, evaluation, and production deployment.

The objective is to develop the ability to **design, implement, optimize, evaluate, and deploy production-grade RAG applications** rather than simply understand theoretical concepts.

---

## 1. RAG From Scratch — LangChain

**Objective:** Understand the complete RAG pipeline and implement its core components from scratch.

**Topics Covered**
- RAG architecture and workflow
- Document loading, preprocessing, and indexing
- Text chunking strategies and embeddings
- Vector similarity search
- Multi-Query Retrieval
- Query decomposition and RAG Fusion
- Step-Back Prompting and HyDE
- Query routing
- RAPTOR and ColBERT

**Resources**
- [LangChain — RAG From Scratch Playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)
- [GitHub Repository — RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch)

**Expected Outcome:** Build a functional RAG pipeline and understand how retrieval strategies influence the relevance and quality of generated responses.

---

## 2. Advanced RAG — Sunny Savita

**Objective:** Improve retrieval performance through advanced search techniques and contextual optimization.

**Topics Covered**
- Vector databases and semantic search
- BM25 and sparse retrieval
- Dense retrieval and hybrid search
- Maximum Marginal Relevance (MMR)
- Self-Query Retrieval
- Parent Document Retrieval
- Sentence-Window Retrieval
- Contextual Compression
- Reranking techniques
- RAG Fusion

**Resource**

[Advanced RAG — Retrieval and Reranking Full Course](https://www.youtube.com/watch?v=_kpxLkH5vY0)

**Expected Outcome:** Implement an advanced retrieval pipeline that combines keyword search, semantic search, and reranking to improve retrieval relevance.

---

## 3. Production RAG — freeCodeCamp

**Objective:** Develop the engineering skills required to build scalable, maintainable, and deployable RAG applications.

**Topics Covered**
- Document ingestion and chunking
- Embeddings and Chroma vector database
- Similarity search and retrieval optimization
- Debugging and performance improvement
- Hybrid search and contextual retrieval
- LangSmith for tracing and observability
- Supabase and PostgreSQL with pgvector
- FastAPI and REST API development
- LangGraph integration
- Agentic RAG
- GraphRAG and multimodal RAG
- Security, scalability, and deployment considerations

**Resource**

[Production RAG with LangChain and Vector Databases — Full Course](https://www.youtube.com/watch?v=mHxLXzYjQRE)

**Expected Outcome:** Build an end-to-end RAG application with persistent storage, an API layer, evaluation capabilities, and deployment readiness.

---

## 4. Advanced Retrieval Pipeline — Venelin Valkov

**Objective:** Understand how advanced retrieval methods can be combined into a single, optimized search pipeline.

**Topics Covered**
- BM25 and PostgreSQL full-text search
- Hybrid retrieval
- Reciprocal Rank Fusion (RRF)
- Hypothetical Document Embeddings (HyDE)
- FlashRank and reranking
- PostgreSQL with pgvector
- Local retrieval pipeline design

**Resource**

[Advanced Retrieval Pipeline — HyDE, Hybrid Search and Reranking](https://www.classcentral.com/course/youtube-advanced-retrieval-pipeline-for-rag-hyde-hybrid-search-reranking-build-100-local-retrieval-525132)

**Expected Outcome:** Build and evaluate a retrieval pipeline that combines multiple search methods to improve the quality of retrieved context.

---

## 5. Recommended Learning Sequence

Follow this sequence to progress systematically from foundational knowledge to production engineering.

```text
RAG Fundamentals — LangChain
            |
            v
Advanced Retrieval — Sunny Savita
            |
            v
Production RAG — freeCodeCamp
            |
            v
Advanced Retrieval — Venelin Valkov
            |
            v
Build, Evaluate and Deploy a RAG Application
```

---

## 6. Complete Technical Roadmap

### Phase 1: RAG Fundamentals
- Document ingestion and PDF processing
- Text cleaning and chunking strategies
- Semantic chunking and recursive character splitting
- Embeddings and vector representations
- Chroma and PostgreSQL with pgvector
- Similarity search and metadata filtering
- Prompt construction and grounded generation

### Phase 2: Advanced Retrieval
- BM25 and dense retrieval
- Hybrid search
- Reciprocal Rank Fusion (RRF)
- Maximum Marginal Relevance (MMR)
- Reranking
- Multi-Query Retrieval
- Parent Document Retrieval
- Sentence-Window Retrieval
- HyDE and RAG Fusion

### Phase 3: Advanced RAG Architectures
- Query routing and decomposition
- Contextual retrieval
- Corrective RAG (CRAG)
- Self-RAG
- GraphRAG
- Multimodal RAG
- Agentic RAG using LangGraph

### Phase 4: Evaluation and Observability
- Retrieval and generation evaluation
- RAGAS evaluation framework
- Faithfulness and answer relevance
- Context precision and context recall
- LangSmith tracing and debugging
- Test datasets and regression testing
- Latency, cost, and retrieval-quality benchmarking

### Phase 5: Production Engineering
- FastAPI and REST APIs
- PostgreSQL and pgvector
- Authentication and authorization
- Docker and containerization
- Caching and asynchronous processing
- Logging and error handling
- Deployment on AWS or another cloud platform
- Monitoring, security, and scalability

---

## 7. Projects to Build

| Project | Key Technical Skills |
|---|---|
| PDF Question-Answering System | Document ingestion, chunking, embeddings, vector search |
| Advanced Hybrid RAG | BM25, dense retrieval, hybrid search, RRF, reranking |
| Multi-Document Research Assistant | Query decomposition, Multi-Query Retrieval, source citations |
| Agentic RAG Application | LangGraph, routing, tool calling, conditional workflows |
| Production-Ready RAG Platform | FastAPI, PostgreSQL/pgvector, evaluation, Docker, deployment |

For every project, maintain a well-structured GitHub repository containing source code, a README, architecture diagrams, installation instructions, evaluation results, and a working demonstration where feasible.

---

## 8. Job-Readiness Criteria

Before applying for RAG, GenAI, or LLM application engineering internships, ensure you can:

- Implement a RAG pipeline independently.
- Explain chunking, embeddings, vector search, and retrieval trade-offs.
- Implement hybrid search and reranking.
- Diagnose retrieval failures and hallucinations.
- Evaluate RAG performance using a test dataset and appropriate metrics.
- Build APIs using FastAPI.
- Store and retrieve embeddings using PostgreSQL with pgvector.
- Design conditional workflows using LangGraph.
- Debug applications using tracing and logging.
- Containerize and deploy applications using Docker.
- Demonstrate at least two substantial projects on GitHub.

---

## Recommended Starting Point

**Start with:** [LangChain — RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch)

**Then progress to:** [Production RAG with LangChain and Vector Databases](https://www.youtube.com/watch?v=mHxLXzYjQRE)

The priority should be implementation over passive course completion. Build a working baseline first, introduce advanced retrieval methods incrementally, measure their impact, and then add evaluation, observability, and deployment.

**Final objective:** Develop the practical skills required to engineer reliable, measurable, and production-ready RAG applications.
