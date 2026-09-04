"""Retriever Interface for Airline Travel Policy.

Encapsulates FAISS similarity search and document formatting.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import VECTORSTORE_DIR, TOP_K_RETRIEVAL
from src.llm import get_embeddings
from src.ingestion import build_vectorstore


class PolicyRetriever:
    """Manages document retrieval from the persisted FAISS vector index."""

    def __init__(self, top_k: int = TOP_K_RETRIEVAL):
        self.top_k = top_k
        self.vectorstore: FAISS = self._load_or_build()

    def _load_or_build(self) -> FAISS:
        """Loads existing vector index or builds it on demand."""
        return build_vectorstore(force_rebuild=False)

    def retrieve(self, query: str) -> List[Document]:
        """Retrieves top_k most similar document chunks for a given query."""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        return retriever.invoke(query)

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Formats a list of retrieved Document objects into a clean contextual string."""
        if not docs:
            return "No documents found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "Unknown Source")
            page = doc.metadata.get("page", 0) + 1  # 1-indexed page
            formatted.append(f"--- [Document {i} | Source: {source} | Page: {page}] ---\n{doc.page_content.strip()}")
        
        return "\n\n".join(formatted)
