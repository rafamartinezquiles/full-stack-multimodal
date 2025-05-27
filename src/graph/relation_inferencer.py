# Import the GPT-based language model wrapper from LangChain
from langchain_openai import ChatOpenAI

# Tool to create prompt templates with dynamic input variables
from langchain.prompts import PromptTemplate

# Used to generate all pairwise combinations of entities
from itertools import combinations

# Load environment variables from .env file
from dotenv import load_dotenv

# Access system environment variables 
import os

# Load environment variables into runtime
load_dotenv()

# Initialize the GPT-4 language model with deterministic behavior 
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define a prompt template for suggesting relationships between two entities
# The model is asked to:
# - Suggest one relationship label 
# - Return "NONE" if no reasonable link can be found
prompt = PromptTemplate(
    input_variables=["a", "b", "type_a", "type_b"],
    template="""
Suggest a relationship label (verb, uppercase, no spaces) that could connect:
A: "{a}" [{type_a}]
B: "{b}" [{type_b}]

Only respond with one label. If no clear link exists, respond with: NONE.
"""
)

# Define the function to infer relationships between all pairs of entities
def infer_relationships(entities):
    relations = []  

    # Generate all unordered pairs of entities 
    for a, b in combinations(entities, 2):
        # Fill the prompt with entity names and types
        prompt_text = prompt.format(
            a=a["name"], b=b["name"], type_a=a["type"], type_b=b["type"]
        )

        # Ask the LLM to suggest a relationship label
        result = llm.invoke(prompt_text).content.strip()

        # If the model responds with a valid label, add it to the relationships list
        if result != "NONE":
            relations.append((a["name"], result, b["name"]))

    # Return all inferred relationships as a list of triples
    return relations
