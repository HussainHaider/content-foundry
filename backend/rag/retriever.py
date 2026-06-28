"""
backend/rag/retriever.py
LangGraph node — called first in the pipeline.
Retrieves brand context relevant to the brief from Qdrant.
Result stored in state['brand_context'] and shared with ALL downstream agents.
"""

import os
from langchain_voyageai import VoyageAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from backend.graph.state import ContentState

COLLECTION_NAME = "brand_content"
TOP_K = 6


def rag_retriever_node(state: ContentState) -> dict:
    """
    Retrieves the TOP_K most relevant brand document chunks.
    Query is constructed from the brief + target audience.
    Returns brand_context (concatenated text) and rag_sources (filenames).
    """
    query = (
        f"Brand style guide, tone of voice, content examples "
        f"for: {state['target_audience']}. Brief: {state['brief']}"
    )

    qdrant_kwargs = {"url": os.environ.get("QDRANT_URL", "http://localhost:6333")}
    if os.environ.get("QDRANT_API_KEY"):
        qdrant_kwargs["api_key"] = os.environ["QDRANT_API_KEY"]
    client = QdrantClient(**qdrant_kwargs)
    embeddings = VoyageAIEmbeddings(
        model="voyage-3",
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    docs = vectorstore.similarity_search(query, k=TOP_K)

    brand_context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )
    rag_sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    return {
        "brand_context": brand_context,
        "rag_sources": rag_sources,
    }
