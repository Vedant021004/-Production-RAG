# 🔗 LangChain Chains

A practical guide to understanding **Chains in LangChain**, starting from a single LLM call and progressing toward multi-step pipelines and RAG systems.

The goal is not to memorize syntax like:

```python
chain = prompt | llm | parser
```

The goal is to understand **what is actually happening between each component**.

---

# 🧠 What is a Chain?

A **chain** is a sequence of operations where the output of one component becomes the input of the next component.

```text
Input
  ↓
Step 1
  ↓
Step 2
  ↓
Step 3
  ↓
Output
```

For example:

```text
User Input
    ↓
PromptTemplate
    ↓
LLM
    ↓
OutputParser
    ↓
Final Answer
```

---

# 🔥 The Core Idea

The most important concept:

> **Output of A → Input of B**

For example:

```text
PromptTemplate
      ↓
      LLM
      ↓
OutputParser
```

The prompt produced by `PromptTemplate` becomes the input for the LLM.

The LLM's output becomes the input for the parser.

---

# 1️⃣ LLM Without a Chain

You can directly call an LLM:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

response = llm.invoke(
    "Explain RAG in one sentence"
)

print(response.content)
```

Flow:

```text
Question
   ↓
LLM
   ↓
AIMessage
```

No chain is being used here.

---

# 2️⃣ PromptTemplate

A `PromptTemplate` is a reusable prompt structure.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Explain {topic} in simple language.",
    input_variables=["topic"]
)
```

Here:

```text
{topic}
```

is a placeholder.

You can provide its value:

```python
final_prompt = prompt.invoke({
    "topic": "RAG"
})
```

Conceptually, the result becomes:

```text
Explain RAG in simple language.
```

---

# 🧩 What are `template` and `input_variables`?

They are parameters provided by `PromptTemplate`.

### `template`

Contains the actual prompt structure.

```python
template="Explain {topic} in simple language."
```

### `input_variables`

Defines the dynamic values required by the template.

```python
input_variables=["topic"]
```

You choose the variable names.

For example:

```python
prompt = PromptTemplate(
    template="Explain {topic} to a {level} student.",
    input_variables=["topic", "level"]
)
```

Then:

```python
prompt.invoke({
    "topic": "RAG",
    "level": "beginner"
})
```

---

# 3️⃣ Connecting Prompt and LLM

Now we can connect the prompt to the LLM.

```python
chain = prompt | llm
```

This means:

```text
Input
  ↓
PromptTemplate
  ↓
Final Prompt
  ↓
LLM
  ↓
AIMessage
```

Run it:

```python
result = chain.invoke({
    "topic": "RAG"
})

print(result.content)
```

---

# 🔗 What Does `|` Mean?

The pipe operator connects LangChain components.

```python
chain = prompt | llm
```

means:

```text
prompt output
     ↓
   llm input
```

And:

```python
chain = prompt | llm | parser
```

means:

```text
prompt output
     ↓
llm input
     ↓
llm output
     ↓
parser input
```

Think of:

```python
A | B | C
```

as:

```text
A → B → C
```

---

# 4️⃣ Output Parser

LLMs commonly return an `AIMessage`.

Example:

```python
response = llm.invoke("Explain RAG")
```

The response contains:

```python
response.content
```

`StrOutputParser` converts the LLM response into a simple string.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
```

You can manually use it:

```python
response = llm.invoke("Explain RAG")

answer = parser.invoke(response)

print(answer)
```

Flow:

```text
AIMessage
    ↓
StrOutputParser
    ↓
String
```

---

# 5️⃣ Complete Chain

Now connect everything:

```python
chain = prompt | llm | parser
```

Complete flow:

```text
Input
  ↓
PromptTemplate
  ↓
Final Prompt
  ↓
LLM
  ↓
AIMessage
  ↓
StrOutputParser
  ↓
String
```

Complete example:

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt = PromptTemplate(
    template="Explain {topic} in simple language.",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({
    "topic": "Vector Database"
})

print(result)
```

---

# 🔬 What Happens Internally?

When you execute:

```python
result = chain.invoke({
    "topic": "Vector Database"
})
```

Conceptually:

```text
{"topic": "Vector Database"}
              ↓
       PromptTemplate
              ↓
"Explain Vector Database in simple language."
              ↓
             LLM
              ↓
          AIMessage
              ↓
      StrOutputParser
              ↓
           String
```

The final value stored in:

```python
result
```

is the parsed string.

---

# 6️⃣ Without `|`

The same chain can be written manually.

```python
final_prompt = prompt.invoke({
    "topic": "Vector Database"
})

response = llm.invoke(final_prompt)

answer = parser.invoke(response)

print(answer)
```

Flow:

```text
prompt.invoke()
      ↓
final_prompt
      ↓
llm.invoke()
      ↓
response
      ↓
parser.invoke()
      ↓
answer
```

---

# 🔥 Why Use a Chain?

You could manually write:

```python
final_prompt = prompt.invoke(...)
response = llm.invoke(final_prompt)
answer = parser.invoke(response)
```

But when the workflow grows:

```text
Prompt
 ↓
LLM
 ↓
Parser
 ↓
Retriever
 ↓
LLM
 ↓
Parser
 ↓
Another operation
```

manual management becomes harder.

A chain lets you compose the workflow:

```python
chain = prompt | llm | parser
```

So the code becomes easier to read and maintain.

---

# 7️⃣ Direct LLM vs PromptTemplate

You don't always need `PromptTemplate`.

### Simple one-time prompt

```python
response = llm.invoke(
    "Explain RAG in simple language."
)
```

### Dynamic / reusable prompt

```python
prompt = PromptTemplate(
    template="Explain {topic} in simple language.",
    input_variables=["topic"]
)
```

Then:

```python
chain = prompt | llm | parser
```

---

# 🧠 When Should You Create a Chain?

Ask yourself:

> **Are multiple components connected sequentially?**

If:

```text
A → B → C
```

then a chain can be useful.

For example:

```text
Prompt → LLM → Parser
```

```python
chain = prompt | llm | parser
```

But if you're simply doing:

```python
llm.invoke("Hello")
```

there is no need to create a chain.

---

# 8️⃣ Multiple Variables

Chains become more useful when multiple pieces of information are dynamic.

```python
prompt = PromptTemplate(
    template="""
    Explain {topic}
    to a {level} student
    in {language}.
    """,
    input_variables=["topic", "level", "language"]
)
```

Then:

```python
chain = prompt | llm | parser
```

Invoke:

```python
result = chain.invoke({
    "topic": "RAG",
    "level": "beginner",
    "language": "Hindi"
})
```

Flow:

```text
topic ────────┐
level ────────┼──→ PromptTemplate
language ─────┘
                    ↓
                   LLM
                    ↓
                  Parser
                    ↓
                  Answer
```

---

# 9️⃣ Sequential Chains

Now we move beyond a simple chain.

Suppose:

```text
Topic
 ↓
Generate Outline
 ↓
Generate Blog
```

The first LLM produces an outline.

The second LLM uses that outline to generate the blog.

```text
Input Topic
    ↓
LLM 1
    ↓
Outline
    ↓
LLM 2
    ↓
Blog
```

This is a sequential workflow.

Example:

```python
outline_prompt = PromptTemplate(
    template="Create an outline for {topic}",
    input_variables=["topic"]
)

blog_prompt = PromptTemplate(
    template="Write a blog using this outline:\n{outline}",
    input_variables=["outline"]
)
```

Conceptually:

```text
topic
 ↓
outline_prompt
 ↓
LLM
 ↓
outline
 ↓
blog_prompt
 ↓
LLM
 ↓
blog
```

This is an important step toward complex LLM workflows.

---

# 🔟 Chains and RAG

Chains become extremely important in RAG.

A simple RAG workflow looks like:

```text
User Question
      ↓
Retriever
      ↓
Relevant Documents
      ↓
PromptTemplate
      ↓
LLM
      ↓
OutputParser
      ↓
Answer
```

A simplified chain:

```text
Retriever
    ↓
Context
    ↓
Prompt
    ↓
LLM
    ↓
Parser
```

Eventually, you'll build:

```text
User
 ↓
Query Transformation
 ↓
Retriever
 ↓
Reranker
 ↓
Context Compression
 ↓
Prompt
 ↓
LLM
 ↓
Parser
 ↓
Answer
```

---

# 🛣️ Learning Roadmap

```text
                    LANGCHAIN CHAINS
                           │
                           ↓
                    Basic LLM Call
                           │
                           ↓
                     PromptTemplate
                           │
                           ↓
                      LLM Chain
                           │
                           ↓
                   Output Parsers
                           │
                           ↓
                 Runnable Composition
                           │
                           ↓
                  Sequential Workflows
                           │
                           ↓
                   Parallel Workflows
                           │
                           ↓
                  Conditional Routing
                           │
                           ↓
                    Retrieval Chains
                           │
                           ↓
                       RAG Chains
                           │
                           ↓
                  Agentic Workflows
                           │
                           ↓
                       LangGraph
```

---

# 🧪 Practice Tasks

### Beginner

* [ ] Create a simple LLM call
* [ ] Create a `PromptTemplate`
* [ ] Use `prompt.invoke()`
* [ ] Use `llm.invoke()`
* [ ] Use `StrOutputParser`
* [ ] Create `prompt | llm`
* [ ] Create `prompt | llm | parser`

### Intermediate

* [ ] Create a two-step LLM workflow
* [ ] Pass output from LLM 1 to LLM 2
* [ ] Use multiple prompt variables
* [ ] Build sequential workflows
* [ ] Build parallel workflows
* [ ] Build conditional workflows

### Advanced

* [ ] Retrieval chain
* [ ] RAG chain
* [ ] Query transformation
* [ ] Reranking
* [ ] Context compression
* [ ] Agentic RAG
* [ ] Chains inside LangGraph

---

# 🎯 Core Concepts to Remember

### `invoke()`

Executes a component.

```python
component.invoke(input)
```

---

### `PromptTemplate`

Creates reusable dynamic prompts.

```python
PromptTemplate(...)
```

---

### `{variable}`

A dynamic placeholder.

```text
Explain {topic}
```

---

### `input_variables`

Defines the values required by the template.

```python
input_variables=["topic"]
```

---

### `StrOutputParser`

Converts the LLM response into a string.

```python
parser = StrOutputParser()
```

---

### `|`

Connects components.

```python
prompt | llm | parser
```

Meaning:

```text
Prompt output
     ↓
LLM input

LLM output
     ↓
Parser input
```

---

# 🚀 Final Mental Model

Don't memorize:

```python
chain = prompt | llm | parser
```

Think:

```text
What is my input?
        ↓
What should happen first?
        ↓
What is the output?
        ↓
Where should that output go?
        ↓
What should happen next?
```

Then construct:

```text
A → B → C → D
```

as:

```python
chain = A | B | C | D
```

That is the fundamental idea behind **LangChain Chains**.

---

## 🔥 The Bigger Picture

Chains are one of the building blocks for modern AI systems:

```text
                  LANGCHAIN
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
       Chains      Tools      Retrievers
          │          │          │
          └──────────┼──────────┘
                     ↓
                 LangGraph
                     ↓
              Agentic Systems
                     ↓
                Production AI
```

**Learn the flow, not the syntax.**
