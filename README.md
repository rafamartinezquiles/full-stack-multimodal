# Multimodal Knowledge Graph & RAG Pipeline for HR Content Understanding
This project implements a multimodal pipeline capable of ingesting text, extracting knowledge, and enabling intelligent search using Retrieval-Augmented Generation (RAG). It uses cutting-edge tools like LangChain, OpenAI, and Neo4j to build a searchable knowledge graph from unstructured documents like employee handbooks.

![](images/app.png)

## Overview and Background

This multimodal system demonstrates an end-to-end pipeline that ingests and processes diverse data formats including text, images, audio, and video. Each input is transformed into unified text using tools like PyMuPDF, Tesseract, and Whisper, with videos further enriched through key frame extraction and captioning via LLaVA. Extracted text is passed through LLMs to identify entities and relationships, which are then structured into a Neo4j knowledge graph using Python-generated Cypher queries.

Once the knowledge graph is built, the system enables two complementary question-answering modes. The first uses a semantic RAG pipeline to retrieve and answer user queries by searching indexed documents. The second leverages graph-based QA by translating natural language questions into Cypher to query Neo4j directly. This combination allows users to gain insights from multimodal data through both unstructured and structured retrieval.

![](images/workflow.png)

## Table of Contents
```
multimodal-rag-kg/
|__ images/
|   |__ app.png
|   |__ workflow.png
|__ data/
|   |__ sdg.mp3
|   |__ sdg.mp4
|   |__ sdg.pdf
|   |__ sdg.png
|__ src/
|   |__ ingestion/
|   |   |__ pdf_loader.py
|   |   |__ audio_loader.py
|   |   |__ frame_extractor.py
|   |   |__ image_loader.py
|   |   |__ video_loader.py
|   |__ extraction/
|   |   |__ entity_extractor.py
|   |__ graph/
|   |   |__ graph_writer.py
|   |   |__ relation_inferencer.py
|   |__ rag/
|   |   |__ graph_qa.py
|   |   |__ vector_indexer.py
|   |__ vision/
|   |   |__ llava_captioner.py
|__ .env
|__ main.py
|__ app.py
|__ README.md
|__ requirements.txt
```

## Getting started

### Installing
The project is deployed in a local machine, so you need to install the next software and dependencies to start working:

1. Create and activate the new virtual environment for the project

```bash
conda create --name full_stack_multimodal 
conda activate full_stack_multimodal
```

2. Clone repository

```bash
git clone https://github.com/rafamartinezquiles/full-stack-multimodal.git
```

3. In the same folder that the requirements are, install the necessary requirements

```bash
cd full-stack-multimodal
pip install -r requirements.txt
```


### Setup
To run this project successfully, follow these setup steps carefully:

#### Configure your .env file
At the root of the project, edit the *.env* file and ensure it includes the following environment variables:

```bash
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

- **OpenAI API Key:** Required for all LLM-based tasks. Get yours from [OpenAI](https://platform.openai.com/api-keys)
- **Neo4j Credentials:** 
    1. Create a free account at [Neo4j Aura](https://login.neo4j.com/u/signup/identifier?state=hKFo2SBIZnpjXzJJZGlCSkY2aHFnVEQ5OWNLcVd4QVZtdGg2VaFur3VuaXZlcnNhbC1sb2dpbqN0aWTZIDN1TkxVWExQWHRDcVVHQXBXcXdyTXZfR2hvcWNUX0pro2NpZNkgV1NMczYwNDdrT2pwVVNXODNnRFo0SnlZaElrNXpZVG8)
    2. Use default cloud provider/region.
    3. Once your instance is created, click “Connect” → “Drivers” to retrieve your URI, username, and password.

#### Install Tesseract (for OCR) - *Windows Only*
If you're on Windows, download and install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) from this official build.
After installation, locate the full path to *tesseract.exe* and update this line inside *src/ingestion/image_loader.py*:

Another thing to do in case you are on Windows is to go to the following [link](https://github.com/UB-Mannheim/tesseract/wiki) and install the Tesseract library on your computer. Once that is done, take note of the location of the *tesseract.exe* application, and inside the *image_loader.py* file located in *src/ingestion*, modify the following line:

```bash
pytesseract.pytesseract.tesseract_cmd = "your_path"
```

#### Install FFmpeg (for audio/video transcription)
Whisper requires FFmpeg to process audio and video files. To install it:

1. Download the latest FFmpeg build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
2. Unzip it *(e.g., to C:\ffmpeg)*.
3. Add the binary path to your system PATH:

```bash
C:\ffmpeg\bin
```

**Steps to add it to system environment variables**

- Search for "Environment Variables" in the Start Menu.
- Click “Edit the system environment variables”.
- In the new window, click “Environment Variables…”.
- Under System Variables, select Path → click Edit → New → add the path above.
- Click OK on all dialogs.

Once done, FFmpeg will be accessible globally by Whisper and other tools.

## Types of use

### Streamlit App
To launch the interactive interface, simply run:

```bash
streamlit run app.py
```

This will open a local web page where you can upload any combination of *PDF, PNG, JPG, MP3, or MP4* files. Once uploaded, the system will automatically:

- Extract and display a text preview from the file
- Identify and list extracted entities and concepts
- Ingest all structured information into the Neo4j knowledge graph

After the ingestion step, you can ask questions using either:

- **Graph QA:** Executes Cypher queries over the Neo4j graph to extract structured knowledge
- **RAG:** Performs semantic search over indexed document content and provides context-rich answers

Each answer will be accompanied by the source document it was derived from, ensuring full traceability.

### Terminal Execution (CLI Mode)
To run the pipeline entirely from the command line, use:

```bash
python main.py
```

This mode is preconfigured to process sample files located in the *data/* folder (*sdg.pdf, sdg.png, sdg.mp3, sdg.mp4*). It performs the full ingestion, entity extraction, graph construction, semantic indexing, and query execution directly in the terminal.

If you wish to use your own files:

- Add your files to the *data/* directory.
- Update the file paths inside *main.py* accordingly to reflect the new filenames.

This approach is ideal for automated batch processing or headless deployments.

## Workflow Explanation

### Data Ingestion Layer
The Data Ingestion Layer is responsible for converting raw, multimodal inputs into unified, machine-readable text formats. This includes extracting text from documents, images, audio, and video content using specialized tools and models. Below is a breakdown of each supported modality and the underlying logic used to process them.

#### PDF & Text Extraction
Textual content from .pdf and .txt files is extracted using PyMuPDF (fitz). The function extract_text_from_pdf() opens the document, iterates through each page, and aggregates all the text content into a single string.

#### Image OCR (PNG, JPG)
Image files are processed using Tesseract OCR, accessed through the pytesseract wrapper. The function *extract_text_from_image()* loads an image with PIL and applies optical character recognition to extract readable text.

#### Audio Transcription (MP3)
Audio files are transcribed using OpenAI's Whisper model, a state-of-the-art speech recognition system. The *extract_text_from_audio()* function loads a pre-trained Whisper model *(e.g., "base")* and applies it to transcribe the given audio file.

#### Video Processing (MP4)
Video files are processed in two stages:

1. Speech Transcription using Whisper (via moviepy for audio extraction).
2. Key Frame Extraction using OpenCV.

These modular ingestion components ensure that no matter the input format — scanned document, spoken audio, narrated video, or raw PDF, the pipeline can extract meaningful text to be passed downstream for entity extraction, knowledge graph construction, and question answering.

### Entity & Relationship Extraction
Once raw text is extracted from any modality, it is passed into a two-stage LLM-powered pipeline to identify named entities and infer relationships between them. This stage transforms unstructured text into a structured representation suitable for graph construction.

#### Named Entity Extraction
Named entities such as people, organizations, locations, dates, and abstract concepts are extracted using OpenAI’s gpt-4 model via LangChain's LLMChain. The model is prompted using a structured template that instructs it to return a list of entities in a strict JSON format. This approach ensures high precision and consistency in entity extraction across modalities.

#### Relationsip Inference
After entity extraction, pairwise relationships are inferred between entities using a second LLM-based process. For every pair of entities, the system prompts the model to suggest a relationship label *(e.g., "FOUNDED", "BASED_IN")*. If no logical connection is detected, the model returns *"NONE"* and the pair is ignored.

This step enriches the graph structure by providing semantic connections between entities, enabling deeper reasoning and graph-based question answering.

### Knowledge Graph Construction
Once entities and their relationships are extracted, they are persistently stored in a Neo4j knowledge graph. This structured graph enables efficient semantic search, reasoning, and graph-based question answering across all processed documents. A dedicated KnowledgeGraph class manages the connection to the Neo4j database and exposes two main methods:

- **add_entities(entities, source_file):**  Each entity is added to the graph as a node labeled Entity and linked to a corresponding Document node via a :MENTIONS relationship. This preserves traceability of all knowledge elements by associating them with their originating document.

- **add_relationships(relationships):** All inferred relationships between entities are created as directional edges. The relationship type is sanitized (converted to uppercase, alphanumeric-safe) to ensure valid Cypher syntax. For each relationship triple (source, relation, target), a MERGE operation links the two entities.

Internally, this is wrapped inside a Python method that uses the Neo4j driver. This design ensures that the graph remains clean, consistent, and queryable, allowing for both Cypher-based queries and high-level semantic retrieval through RAG.

### Graph Question Answering
To complement Retrieval-Augmented Generation, this system supports direct graph-based question answering by translating natural language questions into Cypher queries, allowing semantic reasoning over the structured Neo4j knowledge graph. Using GPT-4 and LangChain, a prompt template guides the LLM to generate valid Cypher queries that align with the graph schema:

- **Nodes:** :Entity (with name and type), and :Document (with filename)
- **Relationships:** :MENTIONS, :ADVOCATES, :MANAGES, etc.

### Multimodal RAG Pipeline
The Multimodal RAG Pipeline enables users to ask natural language questions and receive intelligent, context-aware answers derived from various document formats including text, images, audio, and video. After each document is ingested and converted into raw text, the content is broken down into overlapping chunks to ensure context is preserved across sections. Each chunk is then enriched with metadata such as the source filename, allowing for clear traceability of the information.

These text chunks are embedded using OpenAI's semantic embedding model and indexed within a Chroma vector database. This process transforms unstructured content into a searchable, vectorized representation that captures the meaning of the text rather than relying solely on keywords. The entire vector store is persisted locally, making the system reusable without requiring reprocessing every time.

When a user submits a question, the system performs a semantic similarity search over the indexed content to retrieve the most relevant chunks. These are then presented along with their original document sources, allowing users to understand not only the answer but also where it came from. This hybrid approach combines the interpretability of source-based retrieval with the flexibility of semantic understanding, making it highly effective for navigating complex, multimodal knowledge.