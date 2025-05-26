from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

import os

embedding = OpenAIEmbeddings()

def index_documents(doc_texts: list[str], metadata_list: list[dict]):
    docs = []
    for text, meta in zip(doc_texts, metadata_list):
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.create_documents([text], metadatas=[meta])
        docs.extend(chunks)
    vectorstore = Chroma.from_documents(docs, embedding, persist_directory="chroma_store")
    vectorstore.persist()
    print(f"Indexed {len(docs)} chunks.")

def retrieve_similar(query: str, k=3):
    vectorstore = Chroma(persist_directory="chroma_store", embedding_function=embedding)
    results = vectorstore.similarity_search(query, k=k)
    return results
