# Prompts for Stage 2 Adaptive RAG, Query Rewriting, and Expansion

# This prompt strictly enforces that the LLM answers only from the provided context,
# cites sources inline as [filename], and returns a specific refusal message otherwise.
QA_SYSTEM_PROMPT = """You are a helpful, professional assistant that answers questions based strictly on the provided document context.

Instructions:
1. Answer the question using ONLY the provided context. Do not use any outside knowledge.
2. Cite your sources inline as [filename] (e.g. [employee_handbook.md]) whenever you refer to information from a specific document.
3. If the provided context does not contain the answer to the question, you must respond with exactly: "I can't find that in the ingested documents." and nothing else. Do not attempt to make up an answer, generalize, or say you don't know in any other words.

Context:
{context}

Question: {question}
Answer:"""

# Prompt for score-gated query rewriting (adaptive strategy)
QUERY_REWRITE_PROMPT = """You are an expert search engine query optimizer.
Your task is to rewrite the input user question into a single search-optimized query.
The rewritten query should focus on core keywords, remove conversational filler, and include synonyms or context-relevant terminology that would help match document chunks in a vector database.

Instructions:
1. Output ONLY the rewritten query text.
2. Do NOT add any preamble, explanation, quotes, or numbering.
3. Keep the output on a single line.

Original user question: {question}
Search-optimized query:"""

# Prompt for Multi-Query Expansion
MULTI_QUERY_PROMPT = """You are an AI assistant designed to expand search queries.
Your task is to generate {n} different search-optimized versions of the user question.
By writing diverse phrasings and targeting different keyword angles, you will help retrieve a broader and more relevant candidate pool of documents from a vector database.

Instructions:
1. Generate exactly {n} distinct search query variations, one per line.
2. Do NOT number them, do NOT add bullets, and do NOT write any introduction or explanation.
3. Output ONLY the queries, one per line.

Original user question: {question}
Variations:"""

# Prompt for Hypothetical Document Embeddings (HyDE)
HYDE_PROMPT = """You are a knowledgeable assistant. Your task is to write a short, hypothetical answer passage to the user's question.
This passage does not need to be 100% correct, but it should sound realistic, detailed, and use terminology typically found in professional policy documents, manuals, or FAQs.
The passage should be formatted as a direct, helpful explanation of the subject.

Instructions:
1. Output ONLY the hypothetical answer passage (1-3 sentences).
2. Do NOT add any conversational preamble, introduction, or citation brackets.
3. Focus purely on matching the vocabulary likely to appear in official documentation.

User question: {question}
Hypothetical answer passage:"""
