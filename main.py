from src.ingestion.pdf_loader import extract_text_from_pdf
from src.graph.relation_inferencer import infer_relationships
from src.extraction.entity_extractor import extract_entities
from src.graph.graph_writer import KnowledgeGraph
from dotenv import load_dotenv
import os
import json

load_dotenv()

def main():
    pdf_path = "data/employee_handbook.pdf"
    text = extract_text_from_pdf(pdf_path)[:3000]  # Keep within token limits
    print("Extracted Text:\n", text[:500], "\n---")

    entities_json = extract_entities(text)
    print("Extracted Entities:\n", entities_json)

    # Parse entities if it's a JSON string
    entities = json.loads(entities_json)["entities"]

    # Push to Neo4j
    kg = KnowledgeGraph()
    kg.add_entities(entities, os.path.basename(pdf_path))
    relationships = infer_relationships(entities)
    kg.add_relationships(relationships)
    print("Relationships added to Neo4j")
    kg.close()
    print("Entities written to Neo4j")

if __name__ == "__main__":
    main()

