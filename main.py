from src.ingestion.pdf_loader import extract_text_from_pdf
from src.graph.relation_inferencer import infer_relationships
from src.ingestion.image_loader import extract_text_from_image 
from src.extraction.entity_extractor import extract_entities
from src.graph.graph_writer import KnowledgeGraph
from src.rag.graph_qa import answer_question
from src.rag.vector_indexer import index_documents, retrieve_similar 
from src.ingestion.audio_loader import extract_text_from_audio
from src.ingestion.video_loader import extract_audio_text_from_video
from src.ingestion.frame_extractor import extract_key_frames
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
    else:
        image_text = ""

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
    else:
        audio_text = ""

    video_path = "data/policy_briefing.mp4"
    if os.path.exists(video_path):
        print("\nExtracting text from video audio...")
        video_text = extract_audio_text_from_video(video_path)[:3000]
        print("Video Text:\n", video_text[:500], "\n---")

        video_entities_json = extract_entities(video_text)
        print("Extracted Video Entities:\n", video_entities_json)

        video_entities = json.loads(video_entities_json)["entities"]

        kg = KnowledgeGraph()
        kg.add_entities(video_entities, os.path.basename(video_path))
        video_relationships = infer_relationships(video_entities)
        kg.add_relationships(video_relationships)
        kg.close()
        print("Entities and relationships from video written to Neo4j")

        frame_folder = "data/video_frames"
        frames = extract_key_frames(video_path, frame_folder)
        print(f"\nExtracted {len(frames)} key frames:")
        for path in frames:
            print(f" - {path}")
    else:
        video_text = ""

    print("\nAsking question over graph...")
    question = "What concepts are advocated by HR?"
    answer_question(question)

    print("\nIndexing documents for RAG...")
    docs = [text, image_text, audio_text]
    metas = [
        {"source": "employee_handbook.pdf"},
        {"source": "policy_note.jpg"},
        {"source": "hr_policies.mp3"}
    ]
    index_documents(docs, metas)

    print("\nRunning semantic search...")
    rag_query = "What is the purpose of PMDS and soft skills training?"
    results = retrieve_similar(rag_query)

    for r in results:
        print(f"\nSource: {r.metadata['source']}")
        print(r.page_content[:300], "\n---")

if __name__ == "__main__":
    main()
