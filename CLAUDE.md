# CLAUDE.md — Local Doc Q&A Non-Negotiables and Rules

## 1. Project Non-Negotiables
1. **Golden Dataset Freeze**: `eval/golden_dataset.jsonl` is frozen. Any changes require written human owner approval recorded in active plans. This is enforced by a PreToolUse hook.
2. **Local-Only Stack**: Zero external API calls at runtime for embedding or LLM. All must go to local Ollama on port 11434 (`nomic-embed-text` and `llama3.2:3b`).
3. **No Conversational Memory in Phase 1**: Single-turn Q&A only. No message history is stored or injected.
4. **Secrets & Credentials**: All secrets (such as Cohere API keys in Phase 2) must be retrieved from the environment; never hardcode or log them.
5. **Prompt Caching**: Enable prompt caching on the LLM side to optimize context processing and meet cost/performance targets.
6. **RAG Security**: Treat all ingested/retrieved content as untrusted input. Grounding prompt must resist prompt injections.

## 2. Styling and Engineering Standards
- **Python Code Style**: PEP 8 compliance, clean docstrings, clear type hints where appropriate.
- **Config Management**: All tunables must live in `config.yaml`. No hardcoded magic numbers.
- **Ingestion Idempotency**: Ingestion must be deterministic. Re-uploading a file must replace its existing chunks via hash-based IDs rather than duplicating.
- **Testing**: Every pipeline component must have unit tests covering edge cases (empty files, tables, multi-page PDFs, etc.). Tests must pass before any task is considered complete.

## 3. Test Command
- Run unit tests: `pytest`
- Run evaluation: `python eval/hit_at_k.py --k 3`

## 4. Subagent Delegation Map
- `phase-planner`: Decomposes phase prompts into ordered task plans in `plans/`.
- `rag-orchestrator`: Manages overall coordination, waves, worker splits, and gate sign-offs.
- `pipeline-engineer`: Implements Phase 1 core RAG (ingest, chunk, dense retrieve, prompt, generate).
- `rerank-engineer`: Implements Phase 2 upgrades (over-retrieval, Cohere Rerank, relevance threshold).
- `platform-engineer`: Implements Phase 3 productionization (FastAPI service, rate limits, OTel observability, resilience circuit breakers, CI/CD).
- `eval-engineer`: Owns `eval/golden_dataset.jsonl`, eval harness, metrics, and gate sign-offs.
- `code-reviewer`: Reviews diffs for security, correctness, and style; enforces zero open Blocking findings.
