import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from rag.config import load_config
from rag.ingest import (
    calculate_file_hash,
    load_file_content,
    ingest_file,
    reset_index,
    EmbedModelMismatchError
)

def test_calculate_file_hash():
    """Verify that file hash is deterministic, 12 characters, and changes with content."""
    content1 = b"Hello World"
    content2 = b"Hello World"
    content3 = b"Hello World!"

    hash1 = calculate_file_hash(content1)
    hash2 = calculate_file_hash(content2)
    hash3 = calculate_file_hash(content3)

    assert len(hash1) == 12
    assert hash1 == hash2
    assert hash1 != hash3

def test_load_file_content_text(tmp_path):
    """Verify load_file_content parses a text file correctly."""
    text_path = tmp_path / "test.txt"
    content = "Hello from text file!"
    text_path.write_text(content, encoding="utf-8")

    text, file_bytes = load_file_content(str(text_path))
    assert text == content
    assert file_bytes == content.encode("utf-8")

def test_load_file_content_empty_text(tmp_path):
    """Verify that empty file raises ValueError."""
    text_path = tmp_path / "empty.txt"
    text_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="is empty"):
        load_file_content(str(text_path))

@patch("pypdf.PdfReader")
def test_load_file_content_pdf(mock_pdf_reader, tmp_path):
    """Verify that load_file_content extracts pages from PDF correctly."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"dummy pdf bytes")

    # Mock PdfReader and pages
    mock_reader_inst = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 Content"
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2 Content"

    mock_reader_inst.pages = [mock_page1, mock_page2]
    mock_pdf_reader.return_value = mock_reader_inst

    text, file_bytes = load_file_content(str(pdf_path))
    assert "Page 1 Content" in text
    assert "Page 2 Content" in text
    assert file_bytes == b"dummy pdf bytes"

@patch("rag.ingest.get_chroma_collection_and_embeddings")
def test_ingest_file_replace_semantics(mock_get_store, tmp_path):
    """Verify that ingesting a file deletes old chunks and inserts new prefixed ones."""
    file_path = tmp_path / "sample.txt"
    content = "A simple document chunk. " * 50 # Sized to force multiple chunks
    file_path.write_text(content, encoding="utf-8")

    # Mock Vector Store and Chroma collection
    mock_vector_store = MagicMock()
    mock_collection = MagicMock()
    mock_vector_store._client.get_collection.return_value = mock_collection
    mock_get_store.return_value = (mock_vector_store, MagicMock())

    stats = ingest_file(str(file_path))

    # Assert stats are reported correctly
    assert stats["file"] == "sample.txt"
    assert stats["chunks"] > 1
    assert stats["status"] == "success"

    # Verify Replace Semantics: collection.delete was called for the source filename
    mock_collection.delete.assert_called_once_with(where={"source": "sample.txt"})

    # Verify add_documents was called with prefixed documents and custom ids
    mock_vector_store.add_documents.assert_called()
    call_args = mock_vector_store.add_documents.call_args[0]
    call_kwargs = mock_vector_store.add_documents.call_args[1]

    docs = call_args[0]
    ids = call_kwargs["ids"]

    # Verify search_document prefix is added
    assert all(doc.page_content.startswith("search_document: ") for doc in docs)
    # Verify metadata source is correct
    assert all(doc.metadata["source"] == "sample.txt" for doc in docs)
    # Verify IDs match format hash:index
    file_hash = calculate_file_hash(content.encode("utf-8"))
    assert all(id_val.startswith(f"{file_hash}:") for id_val in ids)

@patch("rag.ingest.PersistentClient")
def test_reset_index(mock_client):
    """Verify that reset_index deletes and recreates the collection."""
    mock_client_inst = MagicMock()
    mock_collection = MagicMock()
    mock_client_inst.list_collections.return_value = [mock_collection]
    mock_collection.name = "docs"
    mock_client.return_value = mock_client_inst

    reset_index()

    mock_client_inst.delete_collection.assert_called_once_with("docs")
    mock_client_inst.create_collection.assert_called_once()

@patch("rag.ingest.PersistentClient")
def test_embed_model_mismatch_validation(mock_client):
    """Verify EmbedModelMismatchError is raised when DB embed model differs from config."""
    mock_client_inst = MagicMock()
    mock_collection = MagicMock()
    mock_collection.name = "docs"
    # Stored model is different from nomic-embed-text
    mock_collection.metadata = {"embed_model": "some-different-embed-model"}
    mock_client_inst.list_collections.return_value = [mock_collection]
    mock_client_inst.get_collection.return_value = mock_collection
    mock_client.return_value = mock_client_inst

    # Loading vector store should raise EmbedModelMismatchError
    from rag.ingest import get_chroma_collection_and_embeddings
    with pytest.raises(EmbedModelMismatchError, match="Embedding model mismatch"):
        get_chroma_collection_and_embeddings()
