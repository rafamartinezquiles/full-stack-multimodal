from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from itertools import combinations
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = PromptTemplate(
    input_variables=["a", "b", "type_a", "type_b"],
    template="""
Suggest a relationship label (verb, uppercase, no spaces) that could connect:
A: "{a}" [{type_a}]
B: "{b}" [{type_b}]

Only respond with one label. If no clear link exists, respond with: NONE.
"""
)

def infer_relationships(entities):
    relations = []
    for a, b in combinations(entities, 2):
        prompt_text = prompt.format(
            a=a["name"], b=b["name"], type_a=a["type"], type_b=b["type"]
        )
        result = llm.invoke(prompt_text).content.strip()
        if result != "NONE":
            relations.append((a["name"], result, b["name"]))
    return relations
