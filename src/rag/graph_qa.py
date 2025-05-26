from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0)
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

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

def question_to_cypher(question: str) -> str:
    return llm.invoke(cypher_template.format(question=question)).content.strip()

def run_cypher_query(cypher: str) -> list:
    with driver.session() as session:
        result = session.run(cypher)
        return [record.data() for record in result]

def answer_question(question: str):
    cypher = question_to_cypher(question)
    print(f"\nCypher:\n{cypher}")

    # ✅ Basic check before running Cypher
    first_word = cypher.strip().split(" ")[0].upper()
    if first_word not in {"MATCH", "CALL", "CREATE", "MERGE", "WITH", "RETURN"}:
        print("GPT did not return a valid Cypher query. Skipping execution.")
        return

    # ✅ Execute safely
    try:
        data = run_cypher_query(cypher)
        print(f"\nResult:\n{data}")
    except Exception as e:
        print(f"Failed to execute Cypher:\n{e}")

