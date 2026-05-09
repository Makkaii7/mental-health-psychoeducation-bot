"""
RAG pipeline: LangChain + ChromaDB + Hugging Face embeddings (langchain-chroma / langchain-huggingface).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Sequence

# Import HuggingFaceEmbeddings before Chroma: on some Windows/Python builds,
# loading chromadb before sentence-transformers/torch can cause a native crash.
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

logger = logging.getLogger(__name__)

_COSINE_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def load_corpus(corpus_dir: str | Path = "data/rag_corpus") -> list[Document]:
    """Load ``.txt`` / ``.md`` from corpus dir. Returns [] with a warning if missing or empty."""
    root = Path(corpus_dir)
    if not root.is_dir():
        logger.warning("RAG corpus directory not found: %s — returning empty corpus", root)
        return []

    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                title = path.stem
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "title": title},
                    )
                )

    if not docs:
        logger.warning(
            "No .txt/.md files under %s — RAG will run without retrieved context until you add corpus files.",
            root,
        )
    return docs


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(list(documents))


def create_vectorstore(
    chunks: Sequence[Document],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: str | Path = "chroma_db",
    collection_name: str = "psychoeducation",
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return Chroma.from_documents(
            documents=list(chunks),
            embedding=embeddings,
            persist_directory=str(persist_directory),
            collection_name=collection_name,
            collection_metadata=_COSINE_COLLECTION_METADATA,
        )


def load_vectorstore(
    persist_directory: str | Path = "chroma_db",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    collection_name: str = "psychoeducation",
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
        collection_metadata=_COSINE_COLLECTION_METADATA,
    )


def retrieve(vectorstore: Chroma, query: str, k: int = 3) -> list[Document]:
    return vectorstore.similarity_search(query, k=k)


def format_context(chunks: Sequence[Document]) -> str:
    """Format chunks for the LLM using document title only (no filesystem paths)."""
    parts: list[str] = []
    for i, doc in enumerate(chunks, start=1):
        title = doc.metadata.get("title")
        if not title and doc.metadata.get("source"):
            title = Path(str(doc.metadata["source"])).stem
        title = title or "untitled"
        parts.append(f"[{i}] (source title: {title})\n{doc.page_content}")
    return "\n\n".join(parts)
