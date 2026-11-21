from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "Because it attempts to preserve natural text boundaries such as paragraphs, " \
       "lines, sentences, and words instead of arbitrarily cutting text at fixed positions."

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(chunks)