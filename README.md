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
|__ notebooks/
|__ src/
|   |__ ingestion/
|   |   |__ pdf_loader.py
|   |__ extraction/
|   |   |__ entity_extractor.py
|   |__ graph/
|   |   |__ graph_writer.py
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

Also, to be able to connect everything to Neo4j, we need to go to the following [link](https://login.neo4j.com/u/signup/identifier?state=hKFo2SBIZnpjXzJJZGlCSkY2aHFnVEQ5OWNLcVd4QVZtdGg2VaFur3VuaXZlcnNhbC1sb2dpbqN0aWTZIDN1TkxVWExQWHRDcVVHQXBXcXdyTXZfR2hvcWNUX0pro2NpZNkgV1NMczYwNDdrT2pwVVNXODNnRFo0SnlZaElrNXpZVG8) to create an account and fill out all the required fields. Once that’s done, we leave the default cloud provider and region as-is, and we’ll gain access to our account. A default instance will be created automatically, and we’ll be provided with a username and password — it’s very important to save these credentials, as they will be needed in our code.

Once the instance is created, go to your instances, click on "Connect", then "Drivers". There you’ll find the third piece of information we need: the URI.

Now replace the three values in your .env file as follows (without quotes):

```bash
NEO4J_URI="your_uri"
NEO4J_USERNAME="your_username"
NEO4J_PASSWORD="your_password"
```

