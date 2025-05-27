# Import OpenAI's Chat LLM wrapper from LangChain
from langchain_openai import ChatOpenAI

# Tool to define prompt templates with placeholders
from langchain.prompts import PromptTemplate

# Import the official Neo4j driver to execute Cypher queries
from neo4j import GraphDatabase

# OS interaction for environment variables
import os

# Load .env file to access Neo4j credentials
from dotenv import load_dotenv

# Load environment variables into runtime
load_dotenv()

# Initialize the GPT-4 language model with deterministic output
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Set up a Neo4j driver using credentials from environment variables
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Define a prompt that instructs the LLM to generate a Cypher query
# based on a user's natural language question about a known graph schema
cypher_template = PromptTemplate(
    input_variables=["question"],
    template="""
You are a Cypher expert for a Neo4j graph with this schema:

- All nodes are labeled :Entity
- Entities have properties: name (string), type (Person | Organization | Concept | Location | Date)
- Documents are labeled :Document with property filename
- Relationships include: MENTIONS, MANAGES, IS_DURATION_OF, ADVOCATES, etc.

Translate the following natural language question into a valid Cypher query. 
Always use :Entity, and filter by type using WHERE when needed.
Respond with ONLY the Cypher query.

Question: "{question}"
"""
)

# Convert a natural language question into a Cypher query using GPT
def question_to_cypher(question: str) -> str:
    return llm.invoke(cypher_template.format(question=question)).content.strip()

# Execute the Cypher query and return all result records as a list of dictionaries
def run_cypher_query(cypher: str) -> list:
    with driver.session() as session:
        result = session.run(cypher)
        return [record.data() for record in result]

# Full pipeline: convert a question, generate Cypher, validate and run it, print results
def answer_question(question: str):
    # Get Cypher query from GPT
    cypher = question_to_cypher(question)
    print(f"\nCypher:\n{cypher}")

    # Basic validation to ensure the query starts with a Cypher keyword
    first_word = cypher.strip().split(" ")[0].upper()
    if first_word not in {"MATCH", "CALL", "CREATE", "MERGE", "WITH", "RETURN"}:
        print("GPT did not return a valid Cypher query. Skipping execution.")
        return

    try:
        # Execute query and print the result
        data = run_cypher_query(cypher)
        print(f"\nResult:\n{data}")
    except Exception as e:
        # Catch and print any runtime errors during query execution
        print(f"Failed to execute Cypher:\n{e}")
