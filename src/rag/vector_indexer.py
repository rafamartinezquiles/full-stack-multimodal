# Import LangChain's character-based text splitter for chunking large texts
from langchain.text_splitter import CharacterTextSplitter

# Import the Chroma vector store implementation
from langchain_community.vectorstores import Chroma

# Import OpenAI's embedding model for vector generation
from langchain_openai import OpenAIEmbeddings

# Import the base Document class used by LangChain to wrap text chunks with metadata
from langchain.docstore.document import Document

# Import os for file path or environment access
import os

# Initialize the embedding model once 
embedding = OpenAIEmbeddings()

# Function to index a list of raw document texts and their associated metadata into Chroma
def index_documents(doc_texts: list[str], metadata_list: list[dict]):
    docs = []  

    # Loop through each document and its corresponding metadata
    for text, meta in zip(doc_texts, metadata_list):
        # Split the document into overlapping text chunks
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents([text], metadatas=[meta])  

        # Add all chunks to the main collection
        docs.extend(chunks)

    # Create or load a Chroma vector store and populate it with embeddings from the chunks
    vectorstore = Chroma.from_documents(docs, embedding, persist_directory="chroma_store")

    # Save the vector store to disk so it can be reused later
    vectorstore.persist()

    # Log how many text chunks were indexed
    print(f"Indexed {len(docs)} chunks.")

# Function to retrieve the top-k most semantically similar documents for a query
def retrieve_similar(query: str, k=3):
    # Load the persisted Chroma vector store and specify the embedding function
    vectorstore = Chroma(persist_directory="chroma_store", embedding_function=embedding)

    # Perform a similarity search for the input query
    results = vectorstore.similarity_search(query, k=k)

    # Return the matched results as a list of Document objects
    return results
