# Import Neo4j driver to connect and run Cypher queries
from neo4j import GraphDatabase

# Standard modules for loading environment variables and working with strings
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file 
load_dotenv()

# Fetch Neo4j credentials and connection URI from environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def sanitize_relationship_type(rel_type: str) -> str:
    """
    Convert raw relationship type strings into a safe, Cypher-compatible format:
    - Uppercase
    - Replaces non-alphanumeric characters with underscores
    - Strips leading/trailing underscores
    """
    return re.sub(r'\W+', '_', rel_type.upper()).strip('_')

class KnowledgeGraph:
    """
    Class to interact with a Neo4j knowledge graph:
    - Connects to Neo4j database
    - Adds entities and their relationships
    - Associates entities with their document of origin
    """

    def __init__(self):
        # Create a Neo4j driver instance using the provided credentials
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def close(self):
        # Gracefully close the Neo4j driver session
        self.driver.close()

    def add_entities(self, entities: list, source_file: str):
        """
        Add extracted entities to the graph and link them to a source document.
        For each entity:
        - Ensure the (Entity) node exists (MERGE)
        - Ensure the (Document) node exists (MERGE)
        - Create a :MENTIONS relationship from the document to the entity
        """
        with self.driver.session() as session:
            for ent in entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name, type: $type})
                    MERGE (d:Document {filename: $filename})
                    MERGE (d)-[:MENTIONS]->(e)
                    """,
                    name=ent["name"],
                    type=ent["type"],
                    filename=source_file
                )

    def add_relationships(self, relationships):
        """
        Add relationships between entities in the graph.
        For each (source, relationship_type, target) tuple:
        - Sanitize the relationship type
        - Match existing source and target entities by name
        - Create or merge a directional relationship between them
        """
        with self.driver.session() as session:
            for source, rel_type, target in relationships:
                safe_rel = sanitize_relationship_type(rel_type) 
                query = f"""
                MATCH (a:Entity {{name: $a}})
                MATCH (b:Entity {{name: $b}})
                MERGE (a)-[r:{safe_rel}]->(b)
                """
                session.run(query, a=source, b=target)
