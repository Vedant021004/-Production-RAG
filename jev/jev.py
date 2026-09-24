import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_typesafe import Choice, TypeSafeClassifier

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

@tool
def calculator(expression: str):
    """Calculate a mathematical expression."""
    return eval(expression)

@tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"The weather in {city} is sunny and 28°C."


jev = TypeSafeClassifier()

user = input("ASK: ")

decision = jev.invoke({
    "state": user,
    "questions": {
        "tool": Choice(
            instructions="Choose which tool should handle this request.",
            criteria={
                "calculator": "Mathematical calculations.",
                "weather": "Weather related questions.",
                "none": "No tool is required."
            }
        )
    }
})

selected_tool = decision.choices["tool"].choice

if selected_tool == "calculator":

    extraction = llm.invoke(
        f"""
        Extract only the mathematical expression from this question.

        Question: {user}

        Return only the expression.
        """
    )

    expression = extraction.content.strip()

    expression = re.sub(r"[^\d+\-*/().% ]", "", expression)

    tool_result = calculator.invoke({
        "expression": expression
    })

elif selected_tool == "weather":

    extraction = llm.invoke(
        f"""
        Extract only the city name from this question.

        Question: {user}

        Return only the city name.
        """
    )

    city = extraction.content.strip()

    tool_result = get_weather.invoke({
        "city": city
    })

else:

    tool_result = "No tool was required."


response = llm.invoke(
    f"""
    User question:
    {user}

    Tool result:
    {tool_result}

    Give the user a clear and natural final answer.
    """
)

print("\nAI:", response.content)