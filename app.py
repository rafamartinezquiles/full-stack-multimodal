import streamlit as st
import tempfile
import os
import json
from PIL import Image

from src.ingestion.pdf_loader import extract_text_from_pdf
from src.ingestion.image_loader import extract_text_from_image
from src.ingestion.audio_loader import extract_text_from_audio
from src.ingestion.video_loader import extract_audio_text_from_video
from src.ingestion.frame_extractor import extract_key_frames
from src.vision.llava_captioner import generate_caption
from src.extraction.entity_extractor import extract_entities
from src.graph.relation_inferencer import infer_relationships
from src.graph.graph_writer import KnowledgeGraph
from src.rag.vector_indexer import index_documents, retrieve_similar
from src.rag.graph_qa import answer_question

st.set_page_config(page_title="Multimodal RAG Graph App", layout="centered")
st.title("Multimodal Knowledge Graph + RAG Explorer")

uploaded_file = st.file_uploader("Upload a PDF, Image, Audio, or Video", type=["pdf", "png", "jpg", "jpeg", "mp3", "mp4"])
rag_texts = []
rag_metas = []

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getbuffer())
        file_path = tmp.name

    ext = uploaded_file.name.split(".")[-1].lower()
    text = ""
    modality = ""
    doc_name = uploaded_file.name

    if ext == "pdf":
        text = extract_text_from_pdf(file_path)[:3000]
        modality = "PDF"

    elif ext in ["jpg", "jpeg", "png"]:
        text = extract_text_from_image(file_path)[:3000]
        modality = "Image"

    elif ext == "mp3":
        text = extract_text_from_audio(file_path)[:3000]
        modality = "Audio"

    elif ext == "mp4":
        text = extract_audio_text_from_video(file_path)[:3000]
        modality = "Video"

        # Frame extraction + LLaVA
        frame_folder = os.path.join("data", "video_frames")
        os.makedirs(frame_folder, exist_ok=True)
        frames = extract_key_frames(file_path, frame_folder)

        st.markdown("**Video key frame captions (LLaVA):**")
        captions = []
        for path in frames:
            try:
                img = Image.open(path).convert("RGB")
                if img.getbbox():
                    caption = generate_caption(path)
                    st.image(path, caption=caption, width=300)
                    captions.append({"name": caption, "type": "Concept"})
            except:
                continue

        if captions:
            kg = KnowledgeGraph()
            kg.add_entities(captions, doc_name)
            rels = infer_relationships(captions)
            kg.add_relationships(rels)
            kg.close()

    if text:
        st.markdown(f"**Extracted Text from {modality}:**")
        st.text_area("Text", text, height=200)

        entities_json = extract_entities(text)
        entities = json.loads(entities_json)["entities"]

        st.markdown("**Extracted Entities:**")
        for ent in entities:
            st.markdown(f"- `{ent['type']}`: {ent['name']}")

        kg = KnowledgeGraph()
        kg.add_entities(entities, doc_name)
        relationships = infer_relationships(entities)
        kg.add_relationships(relationships)
        kg.close()

        rag_texts.append(text)
        rag_metas.append({"source": doc_name})

query = st.text_input("Ask a question (Graph QA or RAG):")

if query:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Graph QA Answer")
        try:
            graph_answer = answer_question(query)
            st.success(graph_answer)
        except Exception as e:
            st.error(f"Graph QA failed: {e}")

    with col2:
        st.markdown("### 🔎 RAG Semantic Answer")
        try:
            if rag_texts:
                index_documents(rag_texts, rag_metas)
            results = retrieve_similar(query)
            for r in results:
                st.markdown(f"**Source**: {r.metadata['source']}")
                st.write(r.page_content[:300])
        except Exception as e:
            st.error(f"RAG failed: {e}")
