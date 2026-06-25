"""
backend/rag/ingest.py

Usage:
    python -m backend.rag.ingest --docs ./brand_docs/

Loads all PDF, TXT, MD files from the given folder.
Chunks them with overlap and stores embeddings in Qdrant.
"""

import os
import argparse
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "brand_content"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 120


def ingest(docs_path: str) -> int:
    """Returns number of chunks ingested."""
    loader = DirectoryLoader(docs_path, glob="**/*.{pdf,txt,md}", show_progress=True)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)

    embeddings = VoyageAIEmbeddings(
        model="voyage-3",
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
    )

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="./brand_docs/")
    args = parser.parse_args()
    n = ingest(args.docs)
    print(f"✅ Ingested {n} chunks into Qdrant")
