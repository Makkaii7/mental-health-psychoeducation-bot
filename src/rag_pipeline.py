"""
RAG pipeline: LangChain + ChromaDB + sentence-transformers embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore


def load_corpus(corpus_dir: str | Path = "data/rag_corpus") -> list[Document]:
    """Load plain-text / markdown psychoeducation documents from ``data/rag_corpus/``."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": str(path)}))
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
    return Chroma.from_documents(
        documents=list(chunks),
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )


def load_vectorstore(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: str | Path = "chroma_db",
    collection_name: str = "psychoeducation",
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def retrieve(vectorstore: Chroma, query: str, k: int = 3) -> list[Document]:
    return vectorstore.similarity_search(query, k=k)


def format_context(chunks: Sequence[Document]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(chunks, start=1):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {src})\n{doc.page_content}")
    return "\n\n".join(parts)
