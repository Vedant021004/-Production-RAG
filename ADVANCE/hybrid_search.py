from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Sample documents
documents = [
    "This is a list which containig sample documents.",
    "Keywords are important for keyword-based search.",
    "Document analysis involves extracting keywords.",
    "Keyword-based search relies on sparse embeddings."
]


query="keyword-based search"


import re
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation

    text = re.sub(r'[^\w\s]', '', text)

    return text

# preprocess_documents=[preprocess_text(doc) for doc in documents]

result = []
for docs in documents:
    result.append(preprocess_text(docs))

print(result)