import os
import uuid
import json
import requests
import fitz
from bs4 import BeautifulSoup
import streamlit as st
from urllib.parse import urlparse
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI


def ReadPDF(filepath: str) -> str:
    with open(filepath,"rb") as file:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for pagenumber in range(pdf.page_count):
            page = pdf.load_page(pagenumber)
            text += page.get_text()
        return text


def splitting_chunks(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    listOfChunks = []
    start = 0
    textlen = len(text)
    end = min(start + chunk_size, textlen)
    while start < textlen:
        single_chunk = text[start:end]
        listOfChunks.append(single_chunk)
        start += chunk_size - overlap
    return listOfChunks


if "qdrant_client" not in st.session_state:
    st.session_state["qdrant_client"] = QdrantClient(url="http:127.0.0.1:6333")
    qdrant_client = st.session_state["qdrant_client"]
    name = "rag"
    collectionInfo = qdrant_client.get_collections().collections
    listOfCollections = []
    for c in collectionInfo:
        if c is not None:
            listOfCollections.append(c.name)
    if name not in listOfCollections:
        qdrant_client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSNE
            )
        )


def embedding_documents(source: str, embedding_client: OpenAI, embedding_model: str = "snowflake-arctic-embed-l-v2.0") -> str:
    if source.lower().endwith(".pdf"):
        base_name = os.path.basename(source)
        file_name = os.path.splitext(base_name)[0]
        text = ReadPDF(source)
    else:
        raise ValueError("Upload only PDF")
    listOfchunks = splitting_chunks(text)
    embedding_vector = embedding_client.embeddings.create(
        model=embedding_model,
        input=listOfchunks
    )
    database_records = []
    for single_chunk_text, single_chunk_embed_data in zip(listOfchunks, embedding_vector):
        unique_identifier_string = str(uuid.uuid4())
        single_chunk_vector = single_chunk_embed_data.embedding
        metadata = {"text": single_chunk_text, "doc_id": file_name}

        single_database_entry = PointStruct(
            id=unique_identifier_string,
            vector=single_chunk_vector,
            payload=metadata
        )
        database_records.append(single_database_entry)

    qdrant_client.upsert(collection_name=name, points=database_records)
    return unique_identifier_string



