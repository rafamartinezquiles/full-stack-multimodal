from src.ingestion.pdf_loader import extract_text_from_pdf
from src.graph.relation_inferencer import infer_relationships
from src.ingestion.image_loader import extract_text_from_image 
from src.extraction.entity_extractor import extract_entities
from src.graph.graph_writer import KnowledgeGraph
from src.rag.graph_qa import answer_question
from src.ingestion.audio_loader import extract_text_from_audio  # ✅ NEW
from dotenv import load_dotenv
import os
import json

load_dotenv()

def main():
    pdf_path = "data/employee_handbook.pdf"
    text = extract_text_from_pdf(pdf_path)[:3000] 
    print("Extracted Text:\n", text[:500], "\n---")

    entities_json = extract_entities(text)
    print("Extracted Entities:\n", entities_json)

    entities = json.loads(entities_json)["entities"]

    kg = KnowledgeGraph()
    kg.add_entities(entities, os.path.basename(pdf_path))
    relationships = infer_relationships(entities)
    kg.add_relationships(relationships)
    print("Relationships added to Neo4j")
    kg.close()
    print("Entities written to Neo4j")

    image_path = "data/policy_note.jpg" 
    if os.path.exists(image_path):
        print("\nExtracting text from image...")
        image_text = extract_text_from_image(image_path)[:3000]
        print("Image Text:\n", image_text[:500], "\n---")

        image_entities_json = extract_entities(image_text)
        print("Extracted Image Entities:\n", image_entities_json)

        image_entities = json.loads(image_entities_json)["entities"]

        kg = KnowledgeGraph()
        kg.add_entities(image_entities, os.path.basename(image_path))
        image_relationships = infer_relationships(image_entities)
        kg.add_relationships(image_relationships)
        kg.close()
        print("Entities and relationships from image written to Neo4j")

    audio_path = "data/hr_policies.mp3"
    if os.path.exists(audio_path):
        print("\nExtracting text from audio...")
        audio_text = extract_text_from_audio(audio_path)[:3000]
        print("Audio Text:\n", audio_text[:500], "\n---")

        audio_entities_json = extract_entities(audio_text)
        print("Extracted Audio Entities:\n", audio_entities_json)

        audio_entities = json.loads(audio_entities_json)["entities"]

        kg = KnowledgeGraph()
        kg.add_entities(audio_entities, os.path.basename(audio_path))
        audio_relationships = infer_relationships(audio_entities)
        kg.add_relationships(audio_relationships)
        kg.close()
        print("Entities and relationships from audio written to Neo4j")

    print("\nAsking question over graph...")
    question = "What concepts are advocated by HR?"
    answer_question(question)

if __name__ == "__main__":
    main()
