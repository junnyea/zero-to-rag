import os
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Generator, Optional
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import cohere

from rag.config import load_config
from rag.ingest import get_chroma_client_and_collection, check_metadata_compatibility
from rag.prompts import (
    QA_SYSTEM_PROMPT,
    QUERY_REWRITE_PROMPT,
    MULTI_QUERY_PROMPT,
    HYDE_PROMPT
)
from rag.tracing import TraceEmitter

# Global cache for local Cross-Encoder (R9)
_LOCAL_RERANKER_MODEL = None

def get_local_reranker():
    """Lazily loads and caches the local cross-encoder reranker model."""
    global _LOCAL_RERANKER_MODEL
    if _LOCAL_RERANKER_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            # ms-marco-MiniLM-L-6-v2: small (~80MB), extremely fast, and highly accurate for CPU reranking
            _LOCAL_RERANKER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            raise RuntimeError(f"Failed to load local cross-encoder model: {e}")
    return _LOCAL_RERANKER_MODEL

def rerank_candidates_local(
    question: str,
    candidates: List[Dict[str, Any]],
    config: Dict[str, Any],
    trace: Optional[TraceEmitter] = None
) -> List[Dict[str, Any]]:
    """
    Reranks document candidates locally using a CrossEncoder model on-device.
    """
    if not candidates:
        return []

    top_k = config.get("top_k", 3)
    local_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    if trace:
        trace.emit(
            phase="REASON",
            step="rerank",
            detail=f"Reranking {len(candidates)} candidate chunks locally using CrossEncoder model '{local_model_name}'",
            payload={"candidate_count": len(candidates), "top_k": top_k, "model": local_model_name}
        )

    if trace:
        trace.emit(
            phase="ACT",
            step="rerank",
            detail="Predicting relevance scores using on-device CrossEncoder model"
        )

    start_time = time.time()
    try:
        model = get_local_reranker()
        # Pair query with each candidate's content
        pairs = [[question, c["content"]] for c in candidates]
        scores = model.predict(pairs)
        duration_ms = int((time.time() - start_time) * 1000)

        # Map scores to candidates
        reranked_results = []
        for idx, score in enumerate(scores):
            float_score = float(score)
            candidate = candidates[idx].copy()
            candidate["rerank_score"] = float_score
            reranked_results.append(candidate)

        # Sort descending by rerank score
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        final_results = reranked_results[:top_k]

        if trace:
            trace.emit(
                phase="OBSERVE",
                step="rerank",
                detail=f"Successfully reranked candidate chunks locally, selected top-{len(final_results)}",
                payload={
                    "reranked_chunks": [
                        {
                            "source": c["source"],
                            "vector_score": round(c["score"], 4),
                            "rerank_score": round(c["rerank_score"], 4),
                            "chunk_index": c["chunk_index"]
                        }
                        for c in final_results
                    ]
                },
                duration_ms=duration_ms
            )

        return final_results

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_msg = str(e)
        if trace:
            trace.emit(
                phase="REASON",
                step="rerank",
                detail=f"Local reranker execution failed: {error_msg}. Falling back to vector similarity scores.",
                payload={"error": error_msg},
                duration_ms=duration_ms
            )
        try:
            import streamlit as st
            st.toast(f"⚠️ Local Reranker failed! {error_msg}. Fell back to vector order.")
        except ImportError:
            pass

        return candidates[:top_k]

def rerank_candidates_cohere(
    question: str,
    candidates: List[Dict[str, Any]],
    config: Dict[str, Any],
    trace: Optional[TraceEmitter] = None
) -> List[Dict[str, Any]]:
    """
    Reranks document candidates using Cohere Rerank API.
    Provides strict graceful fallback to vector score on any error (R3).
    """
    if not candidates:
        return []

    top_k = config.get("top_k", 3)
    rerank_model = config.get("rerank_model", "rerank-v3.5")

    if trace:
        trace.emit(
            phase="REASON",
            step="rerank",
            detail=f"Reranking {len(candidates)} candidate chunks with Cohere model '{rerank_model}'",
            payload={"candidate_count": len(candidates), "top_k": top_k, "model": rerank_model}
        )

    api_key = os.environ.get("COHERE_API_KEY", "").strip()
    if not api_key:
        if trace:
            trace.emit(
                phase="REASON",
                step="rerank",
                detail="Cohere API key missing; falling back to vector ordering",
                payload={"error": "API Key absent"}
            )
        try:
            import streamlit as st
            st.toast("⚠️ Cohere API Key absent! Fell back to vector order.")
        except ImportError:
            pass
        return candidates[:top_k]

    if trace:
        trace.emit(
            phase="ACT",
            step="rerank",
            detail="Calling Cohere Rerank cloud API"
        )

    start_time = time.time()
    try:
        try:
            import streamlit as st
            if "cohere_calls" in st.session_state:
                st.session_state.cohere_calls += 1
        except ImportError:
            pass

        co = cohere.ClientV2(api_key=api_key)
        response = co.rerank(
            model=rerank_model,
            query=question,
            documents=[c["content"] for c in candidates],
            top_n=top_k
        )
        duration_ms = int((time.time() - start_time) * 1000)

        reranked_results = []
        for result in response.results:
            idx = result.index
            score = result.relevance_score
            candidate = candidates[idx].copy()
            candidate["rerank_score"] = score
            reranked_results.append(candidate)

        if trace:
            trace.emit(
                phase="OBSERVE",
                step="rerank",
                detail=f"Successfully reranked candidate chunks, selected top-{len(reranked_results)}",
                payload={
                    "reranked_chunks": [
                        {
                            "source": c["source"],
                            "vector_score": round(c["score"], 4),
                            "rerank_score": round(c["rerank_score"], 4),
                            "chunk_index": c["chunk_index"]
                        }
                        for c in reranked_results
                    ]
                },
                duration_ms=duration_ms
            )

        return reranked_results

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_msg = str(e)
        if trace:
            trace.emit(
                phase="REASON",
                step="rerank",
                detail=f"Cohere Rerank API call failed: {error_msg}. Falling back to vector ordering.",
                payload={"error": error_msg},
                duration_ms=duration_ms
            )
        try:
            import streamlit as st
            st.toast(f"⚠️ Cohere Rerank failed! {error_msg}. Fell back to vector order.")
        except ImportError:
            pass

        return candidates[:top_k]

def rewrite_query_llm(question: str, config: Dict[str, Any], trace: Optional[TraceEmitter] = None) -> str:
    """
    Calls the local LLM to rewrite the user question into a search-optimized query.
    """
    model = config.get("query_llm_model") or config.get("llm_model", "llama3.2:3b")

    if trace:
        trace.emit(
            phase="REASON",
            step="rewrite",
            detail=f"Prompting LLM '{model}' to rephrase query for retrieval optimization",
            payload={"model": model, "original_query": question}
        )
        trace.emit(
            phase="ACT",
            step="rewrite",
            detail="Calling local LLM for query rewriting"
        )

    start_time = time.time()
    try:
        llm = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.0 # Strict focus
        )
        prompt_str = QUERY_REWRITE_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt_str)])
        rewritten = response.content.strip()
        duration_ms = int((time.time() - start_time) * 1000)

        # Clean potential quotes around rewritten query
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
        elif rewritten.startswith("'") and rewritten.endswith("'"):
            rewritten = rewritten[1:-1].strip()

        if trace:
            trace.emit(
                phase="OBSERVE",
                step="rewrite",
                detail=f"Query successfully rewritten to: {rewritten}",
                payload={"rewritten_query": rewritten},
                duration_ms=duration_ms
            )

        return rewritten
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        if trace:
            trace.emit(
                phase="OBSERVE",
                step="rewrite",
                detail=f"Query rewrite failed: {e}. Reverting to original query.",
                payload={"error": str(e)},
                duration_ms=duration_ms
            )
        return question

def generate_multi_queries(question: str, config: Dict[str, Any], trace: Optional[TraceEmitter] = None) -> List[str]:
    """
    Calls the local LLM to generate multiple phrasings (query expansion) of the user question.
    """
    model = config.get("query_llm_model") or config.get("llm_model", "llama3.2:3b")
    n = config.get("multi_query_n", 3)

    if trace:
        trace.emit(
            phase="REASON",
            step="multi_query",
            detail=f"Prompting LLM '{model}' to expand question into {n} variations",
            payload={"model": model, "original_query": question, "count": n}
        )
        trace.emit(
            phase="ACT",
            step="multi_query",
            detail="Calling local LLM for query expansion"
        )

    start_time = time.time()
    try:
        llm = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.0
        )
        prompt_str = MULTI_QUERY_PROMPT.format(question=question, n=n)
        response = llm.invoke([HumanMessage(content=prompt_str)])
        duration_ms = int((time.time() - start_time) * 1000)

        # Parse variants (one per line, filter empty lines)
        variants = []
        for line in response.content.split("\n"):
            line_str = line.strip()
            if not line_str:
                continue
            # Remove bullets, numbers if LLM hallucinated them
            if line_str.startswith("-") or line_str.startswith("*"):
                line_str = line_str[1:].strip()
            elif len(line_str) > 2 and line_str[0].isdigit() and line_str[1] in (".", ")", " "):
                line_str = line_str[line_str.index(line_str[1])+1:].strip()

            # Clean quotes
            if line_str.startswith('"') and line_str.endswith('"'):
                line_str = line_str[1:-1].strip()

            if line_str:
                variants.append(line_str)

        # Keep exactly the top n and original
        variants = variants[:n]

        if trace:
            trace.emit(
                phase="OBSERVE",
                step="multi_query",
                detail=f"Successfully expanded query into {len(variants)} variations",
                payload={"variations": variants},
                duration_ms=duration_ms
            )

        return variants
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        if trace:
            trace.emit(
                phase="OBSERVE",
                step="multi_query",
                detail=f"Query expansion failed: {e}",
                payload={"error": str(e)},
                duration_ms=duration_ms
            )
        return []

def generate_hyde_passage(question: str, config: Dict[str, Any], trace: Optional[TraceEmitter] = None) -> str:
    """
    Calls the local LLM to generate a hypothetical answer passage (HyDE strategy).
    """
    model = config.get("query_llm_model") or config.get("llm_model", "llama3.2:3b")

    if trace:
        trace.emit(
            phase="REASON",
            step="hyde",
            detail=f"Prompting LLM '{model}' to generate hypothetical answer passage for HyDE",
            payload={"model": model, "query": question}
        )
        trace.emit(
            phase="ACT",
            step="hyde",
            detail="Calling local LLM for hypothetical passage generation"
        )

    start_time = time.time()
    try:
        llm = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.0
        )
        prompt_str = HYDE_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt_str)])
        passage = response.content.strip()
        duration_ms = int((time.time() - start_time) * 1000)

        if trace:
            trace.emit(
                phase="OBSERVE",
                step="hyde",
                detail="Hypothetical passage generated successfully",
                payload={"hypothetical_passage": passage},
                duration_ms=duration_ms
            )

        return passage
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        if trace:
            trace.emit(
                phase="OBSERVE",
                step="hyde",
                detail=f"HyDE passage generation failed: {e}. Reverting to original query.",
                payload={"error": str(e)},
                duration_ms=duration_ms
            )
        return question

def _retrieve_single_query(query: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Low-level helper to query Chroma vector index for a single search query (without tracing emitter/saving).
    """
    persist_dir = config.get("persist_dir", "./chroma_db")
    embed_model = config.get("embed_model", "nomic-embed-text")
    top_k = config.get("top_k", 3)
    reranker = config.get("reranker", "none")
    candidate_k = config.get("candidate_k", 20)

    # Search depth governs retrieval
    search_k = candidate_k if reranker != "none" else top_k

    client, collection = get_chroma_client_and_collection(persist_dir)
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    # Check model compatibility
    metadata = collection.metadata or {}
    compatibility = check_metadata_compatibility(config, metadata)
    if compatibility == "mismatch":
        return []

    embeddings = OllamaEmbeddings(
        model=embed_model,
        base_url="http://localhost:11434"
    )

    vector_store = Chroma(
        client=client,
        collection_name="docs",
        embedding_function=embeddings
    )

    prefixed_query = f"search_query: {query}"
    results = vector_store.similarity_search_with_score(prefixed_query, k=search_k)

    retrieved_chunks = []
    for doc, distance in results:
        similarity = max(0.0, min(1.0, 1.0 - distance))
        doc_meta = doc.metadata or {}
        original_text = doc_meta.get("original_text")
        if not original_text:
            original_text = doc.page_content
            if original_text.startswith("search_document: "):
                original_text = original_text[len("search_document: "):]

        retrieved_chunks.append({
            "content": original_text,
            "source": doc_meta.get("source", "unknown"),
            "score": similarity,
            "chunk_index": doc_meta.get("chunk_index", -1)
        })

    retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)
    return retrieved_chunks

def retrieve_context_chunks(question: str, config: Dict[str, Any], trace: Optional[TraceEmitter] = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    Retrieves the top_k chunks matching the question from Chroma, executing swappable strategies and optional reranking.
    Returns:
        - A list of dictionary objects, each representing a chunk:
          {"content": str, "source": str, "score": float, "chunk_index": int}
        - A status message ("ok", "empty_index", or "mismatch")
    """
    persist_dir = config.get("persist_dir", "./chroma_db")
    embed_model = config.get("embed_model", "nomic-embed-text")
    strategy = config.get("retrieval_strategy", "adaptive")
    reranker = config.get("reranker", "none")
    top_k = config.get("top_k", 3)
    candidate_k = config.get("candidate_k", 20)
    trigger_score = config.get("rewrite_trigger_score", 0.5)
    max_rewrites = config.get("max_rewrites", 2)

    # Search depth governs retrieval
    search_k = candidate_k if reranker != "none" else top_k

    if trace:
        trace.emit(
            phase="REASON",
            step="retrieval",
            detail=f"Initiating retrieval strategy '{strategy}' (reranker: '{reranker}', search_k: {search_k})",
            payload={
                "question": question,
                "strategy": strategy,
                "reranker": reranker,
                "search_k": search_k,
                "trigger_score": trigger_score,
                "max_rewrites": max_rewrites
            }
        )

    # 1. Access Chroma client & count checks
    client, collection = get_chroma_client_and_collection(persist_dir)
    total_chunks = collection.count()

    if total_chunks == 0:
        if trace:
            trace.emit(
                phase="OBSERVE",
                step="retrieval",
                detail="Chroma database is empty; retrieval aborted",
                payload={"total_chunks": 0}
            )
        return [], "empty_index"

    # Check for metadata/embedding model mismatch
    metadata = collection.metadata or {}
    compatibility = check_metadata_compatibility(config, metadata)
    if compatibility == "mismatch":
        if trace:
            trace.emit(
                phase="OBSERVE",
                step="retrieval",
                detail="Model mismatch detected; retrieval aborted",
                payload={"config_embed": embed_model, "stored_embed": metadata.get("embed_model")}
            )
        return [], "mismatch"

    # Define variable to store final candidates before reranking
    candidates_pool: List[Dict[str, Any]] = []

    # --- ROUTING RETRIEVAL STRATEGY ---
    if strategy == "plain":
        # Strategy: Standard single query vector search (Stage 1 behavior)
        if trace:
            trace.emit(phase="ACT", step="retrieval", detail=f"Searching Chroma vector index for '{question}'")
        start_time = time.time()
        candidates_pool = _retrieve_single_query(question, config)
        duration_ms = int((time.time() - start_time) * 1000)

        if trace:
            trace.emit(
                phase="OBSERVE",
                step="retrieval",
                detail=f"Vector search returned {len(candidates_pool)} chunks",
                payload={"chunks": [{"source": c["source"], "score": round(c["score"], 4)} for c in candidates_pool]},
                duration_ms=duration_ms
            )

    elif strategy == "adaptive":
        # Strategy: Score-gated adaptive query re-writing (R2)
        if trace:
            trace.emit(phase="ACT", step="retrieval", detail=f"Executing initial vector search for '{question}'")
        start_time = time.time()
        initial_candidates = _retrieve_single_query(question, config)
        duration_ms = int((time.time() - start_time) * 1000)

        top_score = initial_candidates[0]["score"] if initial_candidates else 0.0

        if trace:
            trace.emit(
                phase="OBSERVE",
                step="retrieval",
                detail=f"Initial search returned {len(initial_candidates)} chunks (top score: {top_score:.4f})",
                payload={"chunks": [{"source": c["source"], "score": round(c["score"], 4)} for c in initial_candidates]},
                duration_ms=duration_ms
            )

        if top_score >= trigger_score or max_rewrites <= 0:
            # Score matches trigger or loop disabled -> Early exit
            if trace:
                trace.emit(
                    phase="REASON",
                    step="retrieval",
                    detail=f"Top similarity score {top_score:.4f} >= trigger threshold {trigger_score:.4f}. Early exit.",
                    payload={"top_score": top_score, "threshold": trigger_score}
                )
            candidates_pool = initial_candidates
        else:
            # Below threshold -> Query rewrite loop starts
            if trace:
                trace.emit(
                    phase="REASON",
                    step="retrieval",
                    detail=f"Top similarity score {top_score:.4f} < trigger threshold {trigger_score:.4f}. Starting rewrite loop (max_rewrites: {max_rewrites}).",
                    payload={"top_score": top_score, "threshold": trigger_score, "max_rewrites": max_rewrites}
                )

            candidates_pool = list(initial_candidates)
            current_query = question

            for attempt in range(1, max_rewrites + 1):
                # 1. LLM rephrase query
                rewritten_query = rewrite_query_llm(current_query, config, trace)
                current_query = rewritten_query

                # 2. Search Chroma
                if trace:
                    trace.emit(phase="ACT", step="retrieval", detail=f"Searching Chroma with rewritten query (attempt {attempt}/{max_rewrites})")
                start_time = time.time()
                attempt_candidates = _retrieve_single_query(current_query, config)
                duration_ms = int((time.time() - start_time) * 1000)

                attempt_top_score = attempt_candidates[0]["score"] if attempt_candidates else 0.0

                if trace:
                    trace.emit(
                        phase="OBSERVE",
                        step="retrieval",
                        detail=f"Rewritten query search (attempt {attempt}) returned {len(attempt_candidates)} chunks (top score: {attempt_top_score:.4f})",
                        payload={"attempt": attempt, "top_score": attempt_top_score, "chunks": [{"source": c["source"], "score": round(c["score"], 4)} for c in attempt_candidates]},
                        duration_ms=duration_ms
                    )

                # Add search results into collective pool
                candidates_pool.extend(attempt_candidates)

                # Check trigger
                if attempt_top_score >= trigger_score:
                    if trace:
                        trace.emit(
                            phase="REASON",
                            step="retrieval",
                            detail=f"Attempt {attempt} query achieved similarity score {attempt_top_score:.4f} >= threshold {trigger_score:.4f}. Exiting rewrite loop early.",
                            payload={"attempt": attempt, "top_score": attempt_top_score, "threshold": trigger_score}
                        )
                    break

            # Consolidate candidate pool (union and deduplicate by source and index)
            # R2: proceed with the union (deduplicated) of all candidate pools from every attempt
            seen_chunks = {}
            for chunk in candidates_pool:
                # Key maps uniquely to a document chunk segment
                key = f"{chunk['source']}:{chunk['chunk_index']}"
                if key not in seen_chunks or chunk["score"] > seen_chunks[key]["score"]:
                    seen_chunks[key] = chunk

            deduped_candidates = list(seen_chunks.values())
            # Re-sort descending by score
            deduped_candidates.sort(key=lambda x: x["score"], reverse=True)
            # Cap candidates pool at search_k (candidate_k or top_k)
            candidates_pool = deduped_candidates[:search_k]

            if trace:
                trace.emit(
                    phase="OBSERVE",
                    step="retrieval",
                    detail=f"Union-deduplicated candidate pools across all attempts. Consolidated pool: {len(candidates_pool)} chunks.",
                    payload={"consolidated_chunks": [{"source": c["source"], "score": round(c["score"], 4), "chunk_index": c["chunk_index"]} for c in candidates_pool]}
                )

    elif strategy == "multi_query":
        # Strategy: Multi-Query expansion
        variants = generate_multi_queries(question, config, trace)

        # We search original + all variants
        all_queries = [question] + variants
        all_candidates = []

        for q_idx, q in enumerate(all_queries):
            query_label = f"original query" if q_idx == 0 else f"expanded variant {q_idx}"
            if trace:
                trace.emit(phase="ACT", step="retrieval", detail=f"Searching Chroma vector index for {query_label}: '{q}'")
            start_time = time.time()
            q_candidates = _retrieve_single_query(q, config)
            duration_ms = int((time.time() - start_time) * 1000)

            if trace:
                trace.emit(
                    phase="OBSERVE",
                    step="retrieval",
                    detail=f"Search for '{q_label}' returned {len(q_candidates)} chunks",
                    payload={"query_index": q_idx, "chunks": [{"source": c["source"], "score": round(c["score"], 4)} for c in q_candidates]},
                    duration_ms=duration_ms
                )
            all_candidates.extend(q_candidates)

        # Union and deduplicate
        seen_chunks = {}
        for chunk in all_candidates:
            key = f"{chunk['source']}:{chunk['chunk_index']}"
            if key not in seen_chunks or chunk["score"] > seen_chunks[key]["score"]:
                seen_chunks[key] = chunk

        deduped_candidates = list(seen_chunks.values())
        deduped_candidates.sort(key=lambda x: x["score"], reverse=True)
        # Cap pool size at search_k
        candidates_pool = deduped_candidates[:search_k]

        if trace:
            st_det = f"Consolidated multi-query pool into {len(candidates_pool)} chunks."
            trace.emit(
                phase="OBSERVE",
                step="retrieval",
                detail=st_det,
                payload={"pool_chunks": [{"source": c["source"], "score": round(c["score"], 4)} for c in candidates_pool]}
            )

    elif strategy == "hyde":
        # Strategy: Hypothetical Document Embeddings (HyDE)
        hyde_passage = generate_hyde_passage(question, config, trace)

        if trace:
            trace.emit(phase="ACT", step="retrieval", detail=f"Searching Chroma using hypothetical passage embedding")

        start_time = time.time()
        # Search Chroma using hypothetical passage instead of original question
        candidates_pool = _retrieve_single_query(hyde_passage, config)
        duration_ms = int((time.time() - start_time) * 1000)

        if trace:
            trace.emit(
                phase="OBSERVE",
                step="retrieval",
                detail=f"Chroma search with HyDE returned {len(candidates_pool)} chunks",
                payload={"chunks": [{"source": c["source"], "score": round(c["score"], 4)} for c in candidates_pool]},
                duration_ms=duration_ms
            )

    # --- ROUTING RERANKING STAGE ---
    if reranker == "cohere" and candidates_pool:
        reranked_chunks = rerank_candidates_cohere(question, candidates_pool, config, trace)
        return reranked_chunks, "ok"
    elif reranker == "local" and candidates_pool:
        reranked_chunks = rerank_candidates_local(question, candidates_pool, config, trace)
        return reranked_chunks, "ok"

    # --- FALLBACK / DEFAULT (slice pool to top_k) ---
    return candidates_pool[:top_k], "ok"

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

def ask_question(question: str, config: Dict[str, Any], trace: Optional[TraceEmitter] = None) -> Dict[str, Any]:
    """
    Queries the RAG pipeline synchronously (blocks until answer completes).
    Returns a dictionary conforming to Stage 3:
    {
        "answer": str,
        "contexts": list[str],
        "meta": {
            "status": str, # "ok", "empty_index", "mismatch"
            "retrieved_chunks": list of dict,
            "trace_id": str,
            "trace_events": list of dict
        }
    }
    """
    if not trace:
        trace = TraceEmitter(
            trace_dir=config.get("trace_dir", "./traces"),
            trace_keep=config.get("trace_keep", 200)
        )

    trace.emit(
        phase="REASON",
        step="query",
        detail="Received new user question (sync)",
        payload={
            "question": question,
            "retrieval_strategy": config.get("retrieval_strategy", "plain"),
            "reranker": config.get("reranker", "none"),
            "llm_model": config.get("llm_model", "llama3.2:3b"),
            "embed_model": config.get("embed_model", "nomic-embed-text")
        }
    )

    # 1. Retrieve context
    retrieved_chunks, status = retrieve_context_chunks(question, config, trace)
    if status != "ok":
        trace.emit(
            phase="OBSERVE",
            step="query",
            detail=f"Query aborted: {status}",
            payload={"status": status}
        )
        trace.save_to_disk()
        return {
            "answer": "",
            "contexts": [],
            "meta": {
                "status": status,
                "retrieved_chunks": [],
                "trace_id": trace.trace_id,
                "trace_events": trace.events
            }
        }

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

    trace.emit(
        phase="REASON",
        step="generation",
        detail=f"Prompting LLM model '{llm_model}' with grounded context",
        payload={"model": llm_model, "context_chunks_count": len(retrieved_chunks)}
    )

    trace.emit(
        phase="ACT",
        step="generation",
        detail="Invoking local LLM generation"
    )

    start_time = time.time()
    try:
        response = llm.invoke([HumanMessage(content=prompt_str)])
        answer = response.content
        duration_ms = int((time.time() - start_time) * 1000)

        trace.emit(
            phase="OBSERVE",
            step="generation",
            detail="LLM completed generation successfully",
            payload={"answer_length": len(answer)},
            duration_ms=duration_ms
        )
        trace.save_to_disk()

        return {
            "answer": answer,
            "contexts": [c["content"] for c in retrieved_chunks],
            "meta": {
                "status": "ok",
                "retrieved_chunks": retrieved_chunks,
                "trace_id": trace.trace_id,
                "trace_events": trace.events
            }
        }
    except Exception as e:
        trace.emit(
            phase="OBSERVE",
            step="generation",
            detail=f"LLM generation failed: {e}",
            payload={"error": str(e)}
        )
        trace.save_to_disk()
        raise RuntimeError(f"Ollama generation failed: {e}")

def ask_question_stream(question: str, config: Dict[str, Any], trace: Optional[TraceEmitter] = None) -> Tuple[Generator[str, None, None], List[Dict[str, Any]], str, TraceEmitter]:
    """
    Queries the RAG pipeline and returns a stream (generator) for the answer,
    along with the retrieved chunks, status, and the TraceEmitter instance.
    Returns: (token_generator, retrieved_chunks, status, trace)
    """
    if not trace:
        trace = TraceEmitter(
            trace_dir=config.get("trace_dir", "./traces"),
            trace_keep=config.get("trace_keep", 200)
        )

    trace.emit(
        phase="REASON",
        step="query",
        detail="Received new user question (stream)",
        payload={
            "question": question,
            "retrieval_strategy": config.get("retrieval_strategy", "plain"),
            "reranker": config.get("reranker", "none"),
            "llm_model": config.get("llm_model", "llama3.2:3b"),
            "embed_model": config.get("embed_model", "nomic-embed-text")
        }
    )

    # 1. Retrieve context
    retrieved_chunks, status = retrieve_context_chunks(question, config, trace)
    if status != "ok":
        trace.emit(
            phase="OBSERVE",
            step="query",
            detail=f"Query aborted: {status}",
            payload={"status": status}
        )
        trace.save_to_disk()
        return (g for g in []), [], status, trace

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

    trace.emit(
        phase="REASON",
        step="generation",
        detail=f"Prompting LLM model '{llm_model}' with grounded context (streaming)",
        payload={"model": llm_model, "context_chunks_count": len(retrieved_chunks)}
    )

    trace.emit(
        phase="ACT",
        step="generation",
        detail="Invoking streaming local LLM generation"
    )

    def stream_generator() -> Generator[str, None, None]:
        start_time = time.time()
        try:
            full_answer = []
            for chunk in llm.stream([HumanMessage(content=prompt_str)]):
                if chunk.content:
                    full_answer.append(chunk.content)
                    yield chunk.content

            duration_ms = int((time.time() - start_time) * 1000)
            trace.emit(
                phase="OBSERVE",
                step="generation",
                detail="LLM completed generation successfully",
                payload={"answer_length": len("".join(full_answer))},
                duration_ms=duration_ms
            )
            trace.save_to_disk()
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            trace.emit(
                phase="OBSERVE",
                step="generation",
                detail=f"Ollama streaming generation failed: {e}",
                payload={"error": str(e)},
                duration_ms=duration_ms
            )
            trace.save_to_disk()
            yield f"\n[Error during streaming: {e}]"

    return stream_generator(), retrieved_chunks, "ok", trace
