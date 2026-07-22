import os
import hashlib
from typing import List, Dict, Any, Tuple
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.config import load_config

def get_chroma_client_and_collection(persist_dir: str) -> Tuple[chromadb.PersistentClient, Any]:
    """
    Returns the persistent Chroma client and the raw collection 'docs'.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    # Cosine is the specified distance metric in ADR-4
    collection = client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

def check_metadata_compatibility(config: Dict[str, Any], collection_metadata: Dict[str, Any]) -> str:
    """
    Checks if current config matches the index collection metadata.
    Returns:
        - "mismatch" if embed_model differs (Hard block)
        - "stale" if chunk parameters differ but embed_model matches (Warning)
        - "compatible" if everything matches or index is empty (OK)
    """
    if not collection_metadata:
        return "compatible"

    stored_embed = collection_metadata.get("embed_model")
    if stored_embed is None:
        # No metadata yet
        return "compatible"

    current_embed = config.get("embed_model")
    if stored_embed != current_embed:
        return "mismatch"

    stored_size = collection_metadata.get("chunk_size")
    stored_overlap = collection_metadata.get("chunk_overlap")
    current_size = config.get("chunk_size")
    current_overlap = config.get("chunk_overlap")

    if stored_size != current_size or stored_overlap != current_overlap:
        return "stale"

    return "compatible"

def load_file_content(file_path: str, filename: str) -> str:
    """
    Loads text content from txt, md, or pdf files.
    Throws ValueError for unsupported files or extraction failures.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                raise ValueError(f"Failed to read text file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read text file: {e}")

    elif ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            full_text = "\n\n".join(text_parts).strip()
            if not full_text:
                raise ValueError("PDF contains no extractable text (it might be scanned).")
            return full_text
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")

    else:
        raise ValueError(f"Unsupported file format: {ext}")

def ingest_file(file_content: str, filename: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingests a single file into Chroma:
    1. Splits using RecursiveCharacterTextSplitter.
    2. Prepends Nomic task prefix 'search_document: ' to chunk content.
    3. Computes unique hash-based IDs (sha256(file_bytes)[:12] + ":" + chunk_index).
    4. Deletes any existing chunks for this filename.
    5. Saves into persistent Chroma.
    6. Updates collection metadata with active parameters.
    """
    persist_dir = config.get("persist_dir", "./chroma_db")
    chunk_size = config.get("chunk_size", 800)
    chunk_overlap = config.get("chunk_overlap", 150)
    embed_model = config.get("embed_model", "nomic-embed-text")

    # 1. Initialize client & raw collection
    client, collection = get_chroma_client_and_collection(persist_dir)

    # Check for embedding model mismatch
    metadata = collection.metadata or {}
    compatibility = check_metadata_compatibility(config, metadata)
    if compatibility == "mismatch":
        raise ValueError(
            f"Embedding model mismatch! Index is built with '{metadata.get('embed_model')}', "
            f"but current configuration is '{embed_model}'. Please Reset the Index first."
        )

    # 2. Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    # We will compute the SHA256 of the file content for unique chunk IDs
    file_bytes = file_content.encode("utf-8")
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]

    # Split the text
    chunks = splitter.split_text(file_content)
    if not chunks:
        return {"filename": filename, "chunks_written": 0, "status": "empty"}

    # 3. Create Documents and unique IDs
    documents = []
    chunk_ids = []

    for idx, chunk_text in enumerate(chunks):
        # We attach metadata as defined in 8.1
        # source filename, chunk index, character span
        char_start = file_content.find(chunk_text)
        char_end = char_start + len(chunk_text) if char_start != -1 else -1

        doc_metadata = {
            "source": filename,
            "chunk_index": idx,
            "char_start": char_start,
            "char_end": char_end,
            "file_hash": file_hash
        }

        # Prepend nomic task prefix (ADR-3)
        # nomic-embed-text was trained with 'search_document: ' for ingested chunks
        prefixed_text = f"search_document: {chunk_text}"

        # We store the original chunk_text as a separate field in metadata,
        # or we just let prefixed_text be the main content.
        # If we let prefixed_text be the page_content, when we retrieve it,
        # it will have the 'search_document: ' prefix. This is fine and recommended by ADR-3,
        # or we can store the original text in metadata to cleanly show it in UI.
        # Let's save the original text in metadata to display it cleanly in the UI
        # without the "search_document:" prefix! This is a great user-experience touch!
        doc_metadata["original_text"] = chunk_text

        documents.append(Document(page_content=prefixed_text, metadata=doc_metadata))

        # Chunk ID format: hash:index
        chunk_ids.append(f"{file_hash}:{idx}")

    # 4. Replace Semantics (P0)
    # Deleting all existing chunks for this filename to prevent duplication
    # We query the raw collection for any document matching {"source": filename}
    try:
        # Delete existing chunks by filtering source metadata
        collection.delete(where={"source": filename})
    except Exception:
        # Ignore errors if collection is empty or deletion fails
        pass

    # 5. Initialize LangChain embeddings and Chroma vector store
    embeddings = OllamaEmbeddings(
        model=embed_model,
        base_url="http://localhost:11434"
    )

    vector_store = Chroma(
        client=client,
        collection_name="docs",
        embedding_function=embeddings
    )

    # 6. Add documents to collection
    vector_store.add_documents(documents=documents, ids=chunk_ids)

    # 7. Update collection metadata with current ingest settings (P0)
    new_metadata = dict(collection.metadata or {})
    new_metadata.update({
        "embed_model": embed_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    })
    # Remove hnsw:space to avoid Chroma error on modifying distance function
    if "hnsw:space" in new_metadata:
        del new_metadata["hnsw:space"]
    collection.modify(metadata=new_metadata)

    return {
        "filename": filename,
        "chunks_written": len(documents),
        "status": "success",
        "file_hash": file_hash
    }

def get_index_stats(persist_dir: str) -> Dict[str, Any]:
    """
    Returns statistics about the current persistent collection:
    {
        "total_chunks": int,
        "unique_files": list of str,
        "embed_model": str or None,
        "chunk_size": int or None,
        "chunk_overlap": int or None
    }
    """
    try:
        if not os.path.exists(persist_dir):
            return {
                "total_chunks": 0,
                "unique_files": [],
                "embed_model": None,
                "chunk_size": None,
                "chunk_overlap": None
            }

        client, collection = get_chroma_client_and_collection(persist_dir)
        total_chunks = collection.count()

        unique_files = []
        if total_chunks > 0:
            # Fetch all metadata from collection to aggregate unique files
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            seen_files = set()
            for meta in metadatas:
                if meta and "source" in meta:
                    seen_files.add(meta["source"])
            unique_files = sorted(list(seen_files))

        metadata = collection.metadata or {}

        return {
            "total_chunks": total_chunks,
            "unique_files": unique_files,
            "embed_model": metadata.get("embed_model"),
            "chunk_size": metadata.get("chunk_size"),
            "chunk_overlap": metadata.get("chunk_overlap")
        }
    except Exception:
        return {
            "total_chunks": 0,
            "unique_files": [],
            "embed_model": None,
            "chunk_size": None,
            "chunk_overlap": None
        }

def reset_index(persist_dir: str) -> None:
    """
    Completely deletes the Chroma collection to clear the index safely.
    Avoids physical directory removal which causes SQLite lock corruption in some envs.
    """
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection("docs")
    except Exception:
        pass
