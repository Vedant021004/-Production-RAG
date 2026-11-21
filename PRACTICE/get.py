from langchain_community.document_loaders import TextLoader

# Load TXT file
loader = TextLoader(
    "company.txt",
    encoding="utf-8"
)

# Extract documents
documents = loader.load()

# Access content and metadata
for doc in documents:
    print("CONTENT:")
    print(doc.page_content)

    print("\nMETADATA:")
    print(doc.metadata)

