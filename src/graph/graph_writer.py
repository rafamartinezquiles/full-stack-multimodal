from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import re

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def sanitize_relationship_type(rel_type: str) -> str:
    """Convert relationship labels to Cypher-safe uppercase format"""
    return re.sub(r'\W+', '_', rel_type.upper()).strip('_')

class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def add_entities(self, entities: list, source_file: str):
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
        with self.driver.session() as session:
            for source, rel_type, target in relationships:
                safe_rel = sanitize_relationship_type(rel_type)
                query = f"""
                MATCH (a:Entity {{name: $a}})
                MATCH (b:Entity {{name: $b}})
                MERGE (a)-[r:{safe_rel}]->(b)
                """
                session.run(query, a=source, b=target)
