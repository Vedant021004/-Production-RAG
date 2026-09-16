from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt = PromptTemplate(
    template="Write a summary for the following poem - \n{poem}",
    input_variables=["poem"]
)

parser = StrOutputParser()

loader = TextLoader("data/example.txt")

documents = loader.load()






poem = documents[0].page_content

final_prompt = prompt.invoke({"poem": poem})

response = llm.invoke(final_prompt)

answer = parser.invoke(response)

print(answer)