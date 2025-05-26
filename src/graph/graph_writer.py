from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

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
