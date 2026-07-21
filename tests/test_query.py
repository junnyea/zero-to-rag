import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from rag.query import format_context, query_rag

def test_format_context():
    """Verify that format_context strips 'search_document: ' prefix and formats cleanly."""
    doc1 = Document(
        page_content="search_document: This is chunk 1 content.",
        metadata={"source": "handbook.pdf"}
    )
    doc2 = Document(
        page_content="search_document: This is chunk 2 content.",
        metadata={"source": "faq.md"}
    )

    formatted = format_context([doc1, doc2])

    assert "--- START DOCUMENT: handbook.pdf ---" in formatted
    assert "This is chunk 1 content." in formatted
    assert "search_document:" not in formatted
    assert "--- START DOCUMENT: faq.md ---" in formatted
    assert "This is chunk 2 content." in formatted

@patch("rag.query.get_chroma_collection_and_embeddings")
@patch("rag.query.ChatOllama")
def test_query_rag_execution(mock_chat_ollama, mock_get_store):
    """Verify query_rag flow: prefixes question, retrieves, converts score, and streams."""
    # 1. Mock Chroma similarity search with cosine distance
    mock_vector_store = MagicMock()
    doc_mock = Document(
        page_content="search_document: Ingested chunk text.",
        metadata={"source": "test.txt", "chunk_index": 0, "char_span": "0-20"}
    )
    # Cosine distance = 0.15 (strong similarity, i.e., similarity = 1 - 0.15 = 0.85)
    mock_vector_store.similarity_search_with_score.return_value = [(doc_mock, 0.15)]
    mock_get_store.return_value = (mock_vector_store, MagicMock())

    # 2. Mock ChatOllama stream method
    mock_chat_inst = MagicMock()
    mock_chunk1 = MagicMock(content="Answer ")
    mock_chunk2 = MagicMock(content="from ")
    mock_chunk3 = MagicMock(content="context.")
    mock_chat_inst.stream.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
    mock_chat_ollama.return_value = mock_chat_inst

    # 3. Execute query
    stream_gen, chunks_info = query_rag("What is the answer?")

    # Verify prefix was added to search query
    mock_vector_store.similarity_search_with_score.assert_called_once_with(
        "search_query: What is the answer?",
        k=3
    )

    # Verify chunks info conversion (similarity = 1.0 - 0.15 = 0.85)
    assert len(chunks_info) == 1
    assert chunks_info[0]["content"] == "Ingested chunk text."
    assert chunks_info[0]["source"] == "test.txt"
    assert chunks_info[0]["score"] == 0.85

    # Consume stream and verify content
    streamed_tokens = list(stream_gen)
    assert streamed_tokens == ["Answer ", "from ", "context."]
