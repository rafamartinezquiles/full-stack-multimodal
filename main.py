from src.ingestion.pdf_loader import extract_text_from_pdf  # Extract raw text content from PDF files
from src.graph.relation_inferencer import infer_relationships  # Infer semantic relationships between extracted entities
from src.ingestion.image_loader import extract_text_from_image  # Perform OCR to extract text from image files
from src.extraction.entity_extractor import extract_entities  # Extract structured entities from text
from src.graph.graph_writer import KnowledgeGraph  # Handles writing entities and relationships to Neo4j
from src.rag.graph_qa import answer_question  # Perform question answering using the knowledge graph
from src.rag.vector_indexer import index_documents, retrieve_similar  # RAG indexing and semantic retrieval functions
from src.ingestion.audio_loader import extract_text_from_audio  # Transcribe speech to text from audio files
from src.ingestion.video_loader import extract_audio_text_from_video  # Extract and transcribe audio track from video
from src.ingestion.frame_extractor import extract_key_frames  # Extract representative frames from a video
from src.vision.llava_captioner import generate_caption  # Generate natural language captions from images (frames)

from dotenv import load_dotenv  # Load environment variables from .env
from PIL import Image  # Handle image file operations
import os  # File and path utilities
import json  # Parse and format structured JSON

load_dotenv()  # Load environment variables before processing begins

def main():
    # ---------- 1. PDF INGESTION ----------
    # Load and extract up to 3000 characters from the input PDF document
    pdf_path = "data/employee_handbook.pdf"
    text = extract_text_from_pdf(pdf_path)[:3000]  # Truncate for demo or initial prototyping
    print("Extracted Text:\n", text[:500], "\n---")  # Preview the first 500 characters

    # Use NLP model to extract entities from the PDF text
    entities_json = extract_entities(text)
    print("Extracted Entities:\n", entities_json)
    entities = json.loads(entities_json)["entities"]

    # Initialize Neo4j writer, add entities and inferred relationships, then close the session
    kg = KnowledgeGraph()
    kg.add_entities(entities, os.path.basename(pdf_path))
    relationships = infer_relationships(entities)
    kg.add_relationships(relationships)
    kg.close()
    print("Entities and relationships from PDF written to Neo4j")

    # ---------- 2. IMAGE INGESTION ----------
    image_path = "data/policy_note.jpg" 
    image_text = ""
    if os.path.exists(image_path):
        print("\nExtracting text from image...")
        # Perform OCR on the image and extract up to 3000 characters
        image_text = extract_text_from_image(image_path)[:3000]
        print("Image Text:\n", image_text[:500], "\n---")

        # Extract entities from image-derived text
        image_entities_json = extract_entities(image_text)
        print("Extracted Image Entities:\n", image_entities_json)
        image_entities = json.loads(image_entities_json)["entities"]

        # Write entities and relationships from image to the knowledge graph
        kg = KnowledgeGraph()
        kg.add_entities(image_entities, os.path.basename(image_path))
        image_relationships = infer_relationships(image_entities)
        kg.add_relationships(image_relationships)
        kg.close()
        print("Entities and relationships from image written to Neo4j")

    # ---------- 3. AUDIO INGESTION ----------
    audio_path = "data/hr_policies.mp3"
    audio_text = ""
    if os.path.exists(audio_path):
        print("\nExtracting text from audio...")
        # Transcribe audio into text
        audio_text = extract_text_from_audio(audio_path)[:3000]
        print("Audio Text:\n", audio_text[:500], "\n---")

        # Extract and link entities from the transcribed audio
        audio_entities_json = extract_entities(audio_text)
        print("Extracted Audio Entities:\n", audio_entities_json)
        audio_entities = json.loads(audio_entities_json)["entities"]

        kg = KnowledgeGraph()
        kg.add_entities(audio_entities, os.path.basename(audio_path))
        audio_relationships = infer_relationships(audio_entities)
        kg.add_relationships(audio_relationships)
        kg.close()
        print("Entities and relationships from audio written to Neo4j")

    # ---------- 4. VIDEO INGESTION ----------
    video_path = "data/policy_briefing.mp4"
    video_text = ""
    if os.path.exists(video_path):
        print("\nExtracting text from video audio...")
        # Extract and transcribe audio track from video
        video_text = extract_audio_text_from_video(video_path)[:3000]
        print("Video Text:\n", video_text[:500], "\n---")

        # Extract structured entities from transcribed video content
        video_entities_json = extract_entities(video_text)
        print("Extracted Video Entities:\n", video_entities_json)
        video_entities = json.loads(video_entities_json)["entities"]

        # Write entities and their relationships from video content to graph
        kg = KnowledgeGraph()
        kg.add_entities(video_entities, os.path.basename(video_path))
        video_relationships = infer_relationships(video_entities)
        kg.add_relationships(video_relationships)
        kg.close()
        print("Entities and relationships from video audio written to Neo4j")

        # ---------- 4.1 VIDEO FRAME CAPTIONING ----------
        # Extract static visual context from video for multimodal enrichment
        frame_folder = "data/video_frames"
        frames = extract_key_frames(video_path, frame_folder)
        print(f"\nExtracted {len(frames)} key frames:")
        for path in frames:
            print(f" - {path}")

        print("\nGenerating LLaVA captions for key frames...")
        captions = []
        for path in frames:
            try:
                # Ensure image isn't blank before sending to captioner
                img = Image.open(path).convert("RGB")
                if img.getbbox() is None:
                    print(f"Skipping empty image: {path}")
                    continue

                # Generate descriptive text from image using a vision-language model
                caption = generate_caption(path)
                print(f"Caption for {os.path.basename(path)}: {caption}")
                captions.append({
                    "name": caption,
                    "type": "Concept"
                })
            except Exception as e:
                print(f"Error with {path}: {e}")

        # Enrich graph with visual concepts and inferred semantic relationships
        if captions:
            kg = KnowledgeGraph()
            kg.add_entities(captions, os.path.basename(video_path))

            # Infer connections between visual concepts extracted from frames
            caption_relationships = infer_relationships(captions)
            kg.add_relationships(caption_relationships)

            kg.close()
            print("LLaVA captions and relationships added to Neo4j.")

    # ---------- 5. GRAPH QUESTION ANSWERING ----------
    print("\nAsking question over graph...")
    question = "What concepts are advocated by HR?"  # Example natural-language query
    answer_question(question)  # Executes a graph-based query using LLM or Cypher wrapper

    # ---------- 6. INDEX DOCUMENTS FOR RAG ----------
    print("\nIndexing documents for RAG...")
    # Combine all text sources (PDF, image OCR, audio, video) into vector index
    docs = [text, image_text, audio_text, video_text]
    metas = [
        {"source": "employee_handbook.pdf"},
        {"source": "policy_note.jpg"},
        {"source": "hr_policies.mp3"},
        {"source": "policy_briefing.mp4"}
    ]
    index_documents(docs, metas)

    # ---------- 7. RAG SEMANTIC QUERY ----------
    print("\nRunning semantic search...")
    rag_query = "What is the purpose of PMDS and soft skills training?"
    results = retrieve_similar(rag_query)  # Retrieve semantically similar chunks

    # Show matched content with its corresponding source metadata
    for r in results:
        print(f"\nSource: {r.metadata['source']}")
        print(r.page_content[:300], "\n---")

if __name__ == "__main__":
    main()
