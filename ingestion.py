import sys
import os

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rich.console import Console

from src.config import (
    DATA_DIR,
    VECTORSTORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.llm import get_embeddings

console = Console(highlight=False)


def find_pdf_files(directory: Path) -> List[Path]:
    """Recursively search for all .pdf files in the data directory and subdirectories."""
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return []
    return list(directory.rglob("*.pdf"))


def load_pdf_documents(pdf_files: List[Path]) -> List[Document]:
    """Loads all PDF files into LangChain Document objects."""
    all_documents = []
    for pdf_path in pdf_files:
        console.print(f"[cyan][*] Loading PDF:[/cyan] {pdf_path.name} ({pdf_path})")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        # Enrich metadata
        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name
        all_documents.extend(docs)
        console.print(f"   --> Loaded {len(docs)} pages.")
    return all_documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Splits raw documents into smaller semantic chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    console.print(f"[green][+] Split {len(documents)} document pages into {len(chunks)} text chunks.[/green]")
    return chunks


def build_vectorstore(force_rebuild: bool = False) -> FAISS:
    """Builds and persists the FAISS vector index if not already present."""
    if not force_rebuild and (VECTORSTORE_DIR / "index.faiss").exists():
        console.print(f"[bold green][v] Existing FAISS vector store found at:[/bold green] {VECTORSTORE_DIR}")
        embeddings = get_embeddings()
        return FAISS.load_local(
            folder_path=str(VECTORSTORE_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    pdf_files = find_pdf_files(DATA_DIR)
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{DATA_DIR}' or its subdirectories! "
            f"Please place your airline policy PDF into '{DATA_DIR}' (e.g. data/A/airline_policy.pdf)."
        )

    console.print(f"[bold yellow][*] Building Vector Store from {len(pdf_files)} PDF(s)...[/bold yellow]")
    docs = load_pdf_documents(pdf_files)
    chunks = split_documents(docs)

    embeddings = get_embeddings()
    console.print("[cyan][*] Generating embeddings & building FAISS index...[/cyan]")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    console.print(f"[bold green][v] Vector store successfully saved to:[/bold green] {VECTORSTORE_DIR}\n")
    return vectorstore



if __name__ == "__main__":
    build_vectorstore(force_rebuild=True)
