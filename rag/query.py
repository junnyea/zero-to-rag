from typing import Dict, Any, List, Tuple, Generator
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from rag.config import load_config
from rag.ingest import get_chroma_client_and_collection, check_metadata_compatibility
from rag.prompts import QA_SYSTEM_PROMPT

def retrieve_context_chunks(question: str, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Retrieves the top_k chunks matching the question from Chroma.
    Returns:
        - A list of dictionary objects, each representing a chunk:
          {"content": str, "source": str, "score": float, "chunk_index": int}
        - A status message ("ok", "empty_index", or "mismatch")
    """
    persist_dir = config.get("persist_dir", "./chroma_db")
    embed_model = config.get("embed_model", "nomic-embed-text")
    top_k = config.get("top_k", 3)

    # 1. Access Chroma client
    client, collection = get_chroma_client_and_collection(persist_dir)
    total_chunks = collection.count()

    if total_chunks == 0:
        return [], "empty_index"

    # Check for metadata/embedding model mismatch
    metadata = collection.metadata or {}
    compatibility = check_metadata_compatibility(config, metadata)
    if compatibility == "mismatch":
        return [], "mismatch"

    # 2. Setup embeddings and vector store
    embeddings = OllamaEmbeddings(
        model=embed_model,
        base_url="http://localhost:11434"
    )

    vector_store = Chroma(
        client=client,
        collection_name="docs",
        embedding_function=embeddings
    )

    # 3. Prepend Nomic search_query prefix (ADR-3)
    prefixed_query = f"search_query: {question}"

    # 4. Search with scores
    # similarity_search_with_score returns List[Tuple[Document, float]]
    # Since space is cosine, score is cosine distance (0.0 to 2.0 where 0.0 means identical, 1.0 orthogonal)
    results = vector_store.similarity_search_with_score(prefixed_query, k=top_k)

    retrieved_chunks = []
    for doc, distance in results:
        # Cosine similarity = 1 - cosine distance
        similarity = 1.0 - distance
        # Clamp to [0, 1] for robust display
        similarity = max(0.0, min(1.0, similarity))

        # Retrieve the original text from metadata if present (without 'search_document: ')
        doc_meta = doc.metadata or {}
        original_text = doc_meta.get("original_text")
        if not original_text:
            # Fallback and strip the prefix if it was saved with it
            original_text = doc.page_content
            if original_text.startswith("search_document: "):
                original_text = original_text[len("search_document: "):]

        retrieved_chunks.append({
            "content": original_text,
            "source": doc_meta.get("source", "unknown"),
            "score": similarity,
            "chunk_index": doc_meta.get("chunk_index", -1)
        })

    # Sort chunks by similarity score descending (just to be sure)
    retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)

    return retrieved_chunks, "ok"

def format_context_string(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Formats the list of chunks into a single structured string for the prompt context.
    """
    formatted_parts = []
    for chunk in retrieved_chunks:
        formatted_parts.append(
            f"Source: {chunk['source']}\n"
            f"Content:\n{chunk['content']}\n"
            f"---"
        )
    return "\n\n".join(formatted_parts)

def ask_question(question: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Queries the RAG pipeline synchronously (blocks until answer completes).
    Returns a dictionary:
    {
        "answer": str,
        "status": str, # "ok", "empty_index", "mismatch"
        "retrieved_chunks": list of dict
    }
    """
    # 1. Retrieve context
    retrieved_chunks, status = retrieve_context_chunks(question, config)
    if status != "ok":
        return {"answer": "", "status": status, "retrieved_chunks": []}

    # 2. Formulate context and prompt
    context_str = format_context_string(retrieved_chunks)
    prompt_str = QA_SYSTEM_PROMPT.format(context=context_str, question=question)

    # 3. Run LLM
    llm_model = config.get("llm_model", "llama3.2:3b")
    llm = ChatOllama(
        model=llm_model,
        base_url="http://localhost:11434",
        temperature=0.0 # Strict grounding (ADR-3)
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt_str)])
        answer = response.content
        return {
            "answer": answer,
            "status": "ok",
            "retrieved_chunks": retrieved_chunks
        }
    except Exception as e:
        raise RuntimeError(f"Ollama generation failed: {e}")

def ask_question_stream(question: str, config: Dict[str, Any]) -> Tuple[Generator[str, None, None], List[Dict[str, Any]], str]:
    """
    Queries the RAG pipeline and returns a stream (generator) for the answer,
    along with the retrieved chunks and status.
    Returns: (token_generator, retrieved_chunks, status)
    """
    # 1. Retrieve context
    retrieved_chunks, status = retrieve_context_chunks(question, config)
    if status != "ok":
        return (g for g in []), [], status

    # 2. Formulate context and prompt
    context_str = format_context_string(retrieved_chunks)
    prompt_str = QA_SYSTEM_PROMPT.format(context=context_str, question=question)

    # 3. Run LLM with streaming
    llm_model = config.get("llm_model", "llama3.2:3b")
    llm = ChatOllama(
        model=llm_model,
        base_url="http://localhost:11434",
        temperature=0.0 # Strict grounding (ADR-3)
    )

    def stream_generator() -> Generator[str, None, None]:
        try:
            for chunk in llm.stream([HumanMessage(content=prompt_str)]):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n[Error during streaming: {e}]"

    return stream_generator(), retrieved_chunks, "ok"
