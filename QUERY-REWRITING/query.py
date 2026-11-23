
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


# STATE
class State(BaseModel):
    query: str
    prompt: str = ""
    answer: str = ""


# NODE
def query_node(state: State):

    system_prompt = """
    You are an expert prompt engineer.

    Read the user's query and convert it into
    a clear, detailed and structured prompt.

    Return only the improved prompt.
    """

    res = llm.invoke([
        ("system", system_prompt),
        ("human", state.query)
    ])

    return {
        "prompt": res.content
    }

def node_2(state:State):

    response = llm.invoke(state.prompt)

    return {
        "answer" : response.content

    }

# GRAPH
graph = StateGraph(State)

graph.add_node("first", query_node)
graph.add_node("second", node_2)
graph.add_edge(START, "first")
graph.add_edge("first", "second")
graph.add_edge("second",END)

demo = graph.compile()


# USER INPUT
user = input("ASK I'M YOUR AI ASSISTANT: ")

# INVOKE
result = demo.invoke({
    "query": user
})


print("\nIMPROVED PROMPT:")
print(result["prompt"])
print("\nNEW ANSWER")
print(result["answer"])