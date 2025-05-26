# Multimodal Knowledge Graph & RAG Pipeline for HR Content Understanding
This project is a 72-hour technical challenge that implements a multimodal pipeline capable of ingesting text, extracting knowledge, and enabling intelligent search using Retrieval-Augmented Generation (RAG). It uses cutting-edge tools like LangChain, OpenAI, and Neo4j to build a searchable knowledge graph from unstructured documents like employee handbooks.

## Overview and Background

The aim is to demonstrate a functioning multimodal pipeline that:

- Ingests at least three data modalities: text, images, audio/video (WIP)
- Extracts entities and relationships using LLMs
- Stores data as a knowledge graph (Neo4j)
- Enables semantic question answering via a RAG pipeline

## Table of Contents
```
multimodal-rag-kg/
|__ data/
|   |__ employee_handbook.pdf
|__ embeddings/            
|__ graph/                
|__ notebooks/
|__ src/
|   |__ ingestion/
|   |   |__ pdf_loader.py
|   |__ extraction/
|   |   |__ entity_extractor.py
|__ ui/                   
|__ .env
|__ main.py
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
It is essential to ensure that the .env file located in the root directory of the project contains your personal OpenAI API key. To do this, replace the following line with your own OPENAI_API_KEY, which is required to properly use the OpenAI API.

```bash
OPENAI_API_KEY="your_key"
```

Tambien, para ser capaz de enlazar todo con neo4j, hemos de acudir al siguiente link que se encargara 

