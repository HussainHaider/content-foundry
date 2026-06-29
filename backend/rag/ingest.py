"""
backend/rag/ingest.py

Usage:
    python -m backend.rag.ingest --docs ./brand_docs/

Loads all PDF, TXT, MD files from the given folder.
Chunks them with overlap and stores embeddings in Qdrant.
"""

import os
import argparse
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "brand_content"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def ingest(docs_path: str) -> int:
    """Returns number of chunks ingested."""
    docs = []
    for ext, loader_cls, loader_kwargs in (
        ("**/*.md", TextLoader, {"encoding": "utf-8"}),
        ("**/*.txt", TextLoader, {"encoding": "utf-8"}),
        ("**/*.pdf", PyPDFLoader, {}),
    ):
        docs.extend(
            DirectoryLoader(
                docs_path,
                glob=ext,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
                show_progress=True,
            ).load()
        )

    print(f"Loaded {len(docs)} documents from {docs_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = VoyageAIEmbeddings(
        model="voyage-3",
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
    )

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    except Exception as e:
        status = getattr(getattr(e, "status_code", None), "__int__", lambda: None)()
        if "403" in str(e) or status == 403:
            raise RuntimeError(
                f"Qdrant returned 403 Forbidden at {qdrant_url}. "
                "Set QDRANT_API_KEY in your .env if you are using Qdrant Cloud."
            ) from e
        if "404" in str(e) or "Connection" in type(e).__name__:
            raise RuntimeError(
                f"Cannot reach Qdrant at {qdrant_url}. "
                "Ensure Qdrant is running (docker compose up) or QDRANT_URL is correct."
            ) from e
        raise

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
