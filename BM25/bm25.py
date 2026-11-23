
from rank_bm25 import BM25Okapi

# 1. Knowledge Base
documents = [
    "Python is a programming language",
    "Python is widely used in machine learning",
    "Java is used for backend development",
    "Machine learning uses algorithms to train models",
    "Python is useful for artificial intelligence",
    "Deep learning is a subset of machine learning",
    "Java supports object oriented programming",
    "Python is popular for data science and AI"
]

# 2. Tokenization
tokenized_docs = [
    doc.lower().split()
    for doc in documents
]

# 3. Create BM25 Index
bm25 = BM25Okapi(tokenized_docs)

# 4. Take User Query
query = input("Enter your search query: ")

# 5. Tokenize Query
tokenized_query = query.lower().split()

# 6. Calculate Scores
scores = bm25.get_scores(tokenized_query)

# 7. Rank Documents
ranked_docs = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)

# 8. Display Results
print("\n--- BM25 Search Results ---")

for rank, (doc, score) in enumerate(ranked_docs[:3], start=1):
    print(f"\nRank: {rank}")
    print(f"Document: {doc}")
    print(f"Score: {score:.4f}")
