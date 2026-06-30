"""tests/test_ingest.py — Qdrant collection lifecycle on ingest.

Guards the fix that replaced the destructive `recreate_collection` (which wiped
previously ingested docs on every run) with an additive create-if-missing flow.
"""

from unittest.mock import MagicMock, patch


def _patched_ingest(collection_exists: bool):
    """Run ingest() with all Qdrant/loader/embedding deps mocked.

    Returns the mocked QdrantClient so the test can assert which lifecycle
    methods were called.
    """
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = collection_exists

    with patch("backend.rag.ingest.DirectoryLoader") as loader, patch(
        "backend.rag.ingest.VoyageAIEmbeddings"
    ), patch("backend.rag.ingest.QdrantVectorStore") as vectorstore, patch(
        "backend.rag.ingest.QdrantClient", return_value=mock_client
    ):
        loader.return_value.load.return_value = []  # no docs → 0 chunks
        vectorstore.return_value.add_documents = MagicMock()
        from backend.rag import ingest as ingest_mod

        n = ingest_mod.ingest("./brand_docs/", reset=False)
    return mock_client, n


def test_ingest_creates_collection_when_missing():
    client, n = _patched_ingest(collection_exists=False)
    client.create_collection.assert_called_once()
    client.delete_collection.assert_not_called()
    assert n == 0


def test_ingest_is_additive_when_collection_exists():
    """Repeat ingest must NOT drop or recreate an existing collection."""
    client, _ = _patched_ingest(collection_exists=True)
    client.create_collection.assert_not_called()
    client.delete_collection.assert_not_called()
    # The deprecated destructive call must be gone entirely.
    client.recreate_collection.assert_not_called()


def test_ingest_reset_drops_existing_collection():
    mock_client = MagicMock()
    # Exists before the reset (so it's dropped), gone after the drop (so it's
    # recreated) — mirrors how Qdrant reports the collection across the calls.
    mock_client.collection_exists.side_effect = [True, False]

    with patch("backend.rag.ingest.DirectoryLoader") as loader, patch(
        "backend.rag.ingest.VoyageAIEmbeddings"
    ), patch("backend.rag.ingest.QdrantVectorStore") as vectorstore, patch(
        "backend.rag.ingest.QdrantClient", return_value=mock_client
    ):
        loader.return_value.load.return_value = []
        vectorstore.return_value.add_documents = MagicMock()
        from backend.rag import ingest as ingest_mod

        ingest_mod.ingest("./brand_docs/", reset=True)

    mock_client.delete_collection.assert_called_once()
    mock_client.create_collection.assert_called_once()
