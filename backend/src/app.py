import os
import re
import tempfile
import time
import hashlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from chatbot import Chatbot

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH, override=True)

PARSER = os.getenv("PARSER", "local").lower()
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
CHROMA_PATH = os.getenv("CHROMA_PATH", str(Path(__file__).resolve().parent / "db" / "chroma_db"))
COLLECTION_NAME = f"rag_gemini_{GEMINI_EMBED_MODEL.replace('-', '_')}_{EMBEDDING_DIMENSION}_v1"
GROQ_MODEL = "openai/gpt-oss-120b"


@st.cache_resource
def get_groq_client() -> OpenAI:
    api_key = st.secrets.get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("Set LLM_API_KEY in your .env file for Groq chat generation.")
    return OpenAI(base_url=LLM_BASE_URL, api_key=api_key, max_retries=2, timeout=60)


@st.cache_resource
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in your .env file for Gemini embeddings.")
    return genai.Client(api_key=api_key)


class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.client = get_gemini_client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = self.client.models.embed_content(
                model=GEMINI_EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
            )
            if not response.embeddings:
                raise RuntimeError("Gemini returned no embeddings.")
            vectors.extend([list(item.values) for item in response.embeddings])
        if len(vectors) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} embeddings, received {len(vectors)}.")
        return vectors

    def embed_query(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
        )
        if not response.embeddings:
            raise RuntimeError("Gemini returned no query embedding.")
        return list(response.embeddings[0].values)


@st.cache_resource
def get_embedding_function() -> GeminiEmbeddings:
    return GeminiEmbeddings()

@st.cache_resource
def get_vector_store() -> Chroma:
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )


def ReadPDF(filepath: str) -> List[Tuple[int, str]]:
    from llama_cloud import LlamaCloud
    api_key = os.getenv("LLAMACLOUD_API_KEY")
    if not api_key:
        raise ValueError("PARSER=llamacloud requires LLAMACLOUD_API_KEY in .env.")
    client = LlamaCloud(api_key=api_key)
    file_obj = client.files.create(file=filepath, purpose="parse")
    result = client.parsing.parse(
        file_id=file_obj.id,
        tier=os.getenv("PARSE_TIER", "cost_effective"),
        version="latest",
        expand=["markdown_full"],
    )
    return [(0, result.markdown_full or "")]


def splitting_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    listOfChunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        single_chunk = text[start:start + chunk_size].strip()
        if single_chunk:
            listOfChunks.append(single_chunk)
        start += step
    return listOfChunks

def ingest_pdf(filepath: str, original_name: str, progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, float]:
    if not filepath.lower().endswith(".pdf"):
        raise ValueError("Upload only PDF files.")
    timings: Dict[str, float] = {}
    start = time.time()
    pages = ReadPDF(filepath)
    timings["parse"] = time.time() - start
    chunks = []
    for page_no, page_text in pages:
        for idx, chunk_text in enumerate(splitting_chunks(page_text)):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": original_name,
                    "page": int(page_no),
                    "chunk": int(idx),
                    "page_text": page_text.strip() if idx == 0 else "",
                },
            })
    if not chunks:
        raise ValueError("No text could be extracted. If this is a scanned PDF, set PARSER=llamacloud and configure LLAMACLOUD_API_KEY.")
    timings["chunks"] = len(chunks)
    vectorstore = get_vector_store()
    try:
        existing = vectorstore.get(include=["metadatas"])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            vectorstore.delete(ids=existing_ids)
    except Exception:
        pass
    start = time.time()
    total = len(chunks)
    batch_size = 32
    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        ids = [f"{hashlib.sha256((original_name + str(batch_start + i) + item['text']).encode()).hexdigest()}" for i, item in enumerate(batch)]
        vectorstore.add_texts(
            texts=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
            ids=ids,
        )
        if progress:
            progress(min(batch_start + len(batch), total), total)
    timings["embed_and_store"] = time.time() - start
    return timings


def has_document() -> bool:
    try:
        return get_vector_store()._collection.count() > 0
    except Exception:
        return False


def retrieve_context(user_prompt: str, top_k: int = 6) -> str:
    vectorstore = get_vector_store()
    match = re.search(r"\bpages?\s*(\d+)\b", user_prompt, flags=re.IGNORECASE)
    if match:
        page_no = int(match.group(1))
        result = vectorstore.get(where={"page": page_no}, include=["documents", "metadatas"])
        if result.get("ids"):
            rows = sorted(zip(result["documents"], result["metadatas"]), key=lambda row: row[1].get("chunk", 0))
            full_page = next((metadata.get("page_text") for _, metadata in rows if metadata.get("page_text")), "")
            text = full_page or "\n".join(document for document, _ in rows)
            return f"[page {page_no}]\n{text}"
    total = vectorstore._collection.count()
    if total == 0:
        return ""
    overview_pattern = re.compile(r"summar|main point|key point|key takeaway|overview|what is (this|the) .*about|explain (the |this )?(pdf|document|paper|file|book)|review", re.I)
    if overview_pattern.search(user_prompt):
        all_data = vectorstore.get(include=["documents", "metadatas"])
        rows = sorted(zip(all_data["documents"], all_data["metadatas"]), key=lambda row: (row[1].get("page", 0), row[1].get("chunk", 0)))
        if len(rows) > 14:
            step = (len(rows) - 1) / 13
            rows = [rows[round(i * step)] for i in range(14)]
        return "\n\n".join(f"[page {metadata.get('page', '?')}]\n{document}" for document, metadata in rows)
    results = vectorstore.similarity_search(user_prompt, k=top_k)
    return "\n\n".join(f"[page {doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in results)



def main():
    st.set_page_config(
        page_title="Chat with your PDF",
        page_icon="📄",
        layout="wide"
    )
    st.title("Chat with your PDF")
    st.caption(
        f"Embeddings: Gemini ({GEMINI_EMBED_MODEL}, "
        "Vector store: ChromaDB | Chat: Groq"
    )
    if "chatbot" not in st.session_state:
        try:
            get_groq_client()
            st.session_state["chatbot"] = Chatbot(
                model_name=GROQ_MODEL,
                client=get_groq_client(),
                retrieve_context=retrieve_context
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()
    chatbot = st.session_state["chatbot"]
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )
    if uploaded_file is not None:
        key = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()
        if st.session_state.get("ingest_key") != key:
            try:
                with st.spinner(
                    "Extracting PDF..."
                ):
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        suffix_path = temp_file.name
                    try:
                        bar = st.progress(
                            0.0,
                            text="Processing PDF..."
                        )
                        def progress(done: int, total: int):
                            bar.progress(
                                done / total,
                                text=f"{done / total:.0%}"
                            )
                        timings = ingest_pdf(
                            suffix_path,
                            uploaded_file.name,
                            progress
                        )
                        bar.empty()
                    finally:
                        if os.path.exists(suffix_path):
                            os.remove(suffix_path)
                st.session_state["ingest_key"] = key
                st.session_state.pop("answer", None)
                st.success(
                    f"PDF processed successfully"
                )
            except Exception as error:
                st.error(f"PDF ingestion failed: {error}")
    st.divider()
    st.subheader("Ask a question about your PDF")
    if not has_document():
        st.info("Upload and process a PDF to begin.")
        return
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Your question")
        submitted = st.form_submit_button("Get Answer")
    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Generating answer..."):
                    answer, timings = chatbot.chat(question.strip())
                st.session_state["answer"] = answer
                st.session_state["last_context"] = chatbot.last_context
                st.session_state["timings"] = timings
            except Exception as error:
                st.error(f"Request failed: {error}")
    if "answer" in st.session_state:
        st.subheader("Answer")
        st.write(st.session_state["answer"])
if __name__ == "__main__":
    main()
