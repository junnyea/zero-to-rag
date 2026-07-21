import os
import hashlib
import pypdf
from typing import List, Dict, Any, Tuple
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rag.config import load_config

class EmbedModelMismatchError(ValueError):
    """Exception raised when the configured embed model differs from the collection's stored embed model."""
    pass

def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate the first 12 characters of the SHA256 hash of file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()[:12]

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file page by page using pypdf."""
    try:
        reader = pypdf.PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF file '{file_path}': {e}")

def load_file_content(file_path: str) -> Tuple[str, bytes]:
    """Read a file (.txt, .md, .pdf) and return its text content and raw bytes."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path.lower())

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    if ext in [".txt", ".md"]:
        try:
            text_content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                text_content = file_bytes.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Failed to decode text file '{file_path}': {e}")
    elif ext == ".pdf":
        text_content = extract_text_from_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: .txt, .md, .pdf")

    if not text_content.strip():
        raise ValueError(f"Extracted text content from '{file_path}' is empty.")

    return text_content, file_bytes

def get_chroma_collection_and_embeddings():
    """Initialize PersistentClient and OllamaEmbeddings, and verify metadata matches."""
    config = load_config()

    # Initialize OllamaEmbeddings with local URL
    embeddings = OllamaEmbeddings(
        model=config.embed_model,
        base_url="http://localhost:11434"
    )

    client = PersistentClient(path=config.persist_dir)

    # We use a standard collection name 'docs'
    collection_name = "docs"

    # Check if collection exists
    existing_collections = [c.name for c in client.list_collections()]

    if collection_name in existing_collections:
        # Get existing collection to check metadata
        collection = client.get_collection(collection_name)
        metadata = collection.metadata or {}

        stored_embed = metadata.get("embed_model")
        if stored_embed and stored_embed != config.embed_model:
            raise EmbedModelMismatchError(
                f"Embedding model mismatch! Configured model is '{config.embed_model}', "
                f"but the collection is indexed with '{stored_embed}'. "
                f"Please reset and re-index the collection."
            )

    # Initialize the LangChain Chroma wrapper
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={
            "hnsw:space": "cosine",
            "embed_model": config.embed_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap
        }
    )

    return vector_store, embeddings

def ingest_file(file_path: str) -> Dict[str, Any]:
    """Load, split, embed, and upsert a file into the persistent vector database.

    Args:
        file_path: Path to the target file.

    Returns:
        A dict containing ingestion stats (file, chunks, status).
    """
    config = load_config()
    filename = os.path.basename(file_path)

    # Load content and compute hash
    text, file_bytes = load_file_content(file_path)
    file_hash = calculate_file_hash(file_bytes)

    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True
    )

    # Split text
    chunks = splitter.split_text(text)

    # Initialize Vector Store
    vector_store, _ = get_chroma_collection_and_embeddings()

    # Replace Semantics: Delete existing chunks for this filename
    # Chroma allows deleting by metadata using standard client collection
    collection = vector_store._client.get_collection("docs")
    collection.delete(where={"source": filename})

    # Create LangChain Document objects with prefixes and metadata
    documents = []
    chunk_ids = []

    for i, chunk_text in enumerate(chunks):
        # nomic-embed-text requirement: prepend 'search_document: ' task prefix
        prefixed_text = f"search_document: {chunk_text}"

        # Calculate start index and char span
        # Recursive splitter metadata provides start_index
        start_char = text.find(chunk_text)
        # Fallback if find fails for some reason
        if start_char == -1:
            start_char = 0
        end_char = start_char + len(chunk_text)

        doc_metadata = {
            "source": filename,
            "chunk_index": i,
            "char_span": f"{start_char}-{end_char}"
        }

        doc = Document(page_content=prefixed_text, metadata=doc_metadata)
        documents.append(doc)

        # Unique chunk ID: sha256(bytes)[:12] + ":" + chunk_index
        chunk_id = f"{file_hash}:{i}"
        chunk_ids.append(chunk_id)

    # Upsert in batches of 32 for safety and progress
    batch_size = 32
    for j in range(0, len(documents), batch_size):
        batch_docs = documents[j : j + batch_size]
        batch_ids = chunk_ids[j : j + batch_size]
        vector_store.add_documents(batch_docs, ids=batch_ids)

    # Also record model and chunk sizes in the collection metadata
    # The LangChain Chroma wrapper auto-merges or sets it on creation,
    # but we can explicitly update it on the client collection for safety
    collection.modify(metadata={
        "embed_model": config.embed_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap
    })

    return {
        "file": filename,
        "chunks": len(documents),
        "status": "success"
    }

def reset_index() -> None:
    """Completely clear and reset the vector database index."""
    config = load_config()
    client = PersistentClient(path=config.persist_dir)
    collection_name = "docs"

    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        client.delete_collection(collection_name)

    # Re-create empty collection with cosine space
    client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embed_model": config.embed_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap
        }
    )
