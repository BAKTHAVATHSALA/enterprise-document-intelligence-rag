# Codebase Audit Report — Enterprise Document Intelligence & Hybrid RAG Platform

> **Audit Notice**: This document presents a comprehensive audit of the backend codebase against the 17 system requirements outlined in [`doubts.md`](file:///d:/Distributed%20RAG/doubts.md). No source code or implementation files were modified.

---

## A. Complete Project / File Tree

```text
d:\Distributed RAG
├── data_access/
│   ├── __init__.py              # Re-exports data access functions
│   ├── bm25_data_access.py      # BM25 keyword inverted index & exact-term search
│   ├── graph_data_access.py     # Neo4j graph entity-relationship multi-hop search
│   └── vector_data_access.py    # Vector store embeddings & similarity search
├── eval/
│   ├── __init__.py              # Re-exports evaluation functions
│   └── evaluator.py             # Precision@K, Recall@K, MRR & latency evaluator
├── helpers/
│   ├── __init__.py              # Re-exports helper functions
│   ├── citation_helper.py       # Citation extraction & claim validation engine
│   ├── document_helper.py       # Ingestion & layout parsing orchestrator
│   ├── generation_helper.py     # Grounded LLM answer generator
│   └── retrieval_helper.py      # Hybrid multi-retriever search, RRF & reranking
├── interfaces/
│   ├── __init__.py              # Re-exports interface models
│   ├── config_interface.py      # Application config schema
│   ├── document_interface.py    # Document, Chunk, PII, Entity schemas
│   ├── query_interface.py       # Query request/response & citation schemas
│   └── retrieval_interface.py   # Retrieval candidate & RRF schemas
├── services/
│   ├── __init__.py              # Re-exports service functions
│   ├── document_service.py      # Ingestion job coordination & status tracking
│   └── query_service.py         # End-to-end RAG query execution & prompt injection defense
├── tests/
│   └── test_pipeline.py         # Automated test suite
├── utils/
│   ├── __init__.py              # Re-exports utility functions
│   ├── chunker.py               # Layout-aware semantic chunking utility
│   ├── config.py                # Environment configuration loader
│   ├── entity_extractor.py      # Named entity extraction utility
│   ├── logger.py                # Structured logger utility ([funcName] - message)
│   └── pii_redactor.py          # PII detection & masking utility
├── application.md               # Product requirements & architecture specification
├── doubts.md                    # Audit request specification
├── main.py                      # FastAPI REST application entry point
├── README.md                    # System documentation
├── rules.md                     # Coding rules & standards specification
└── run_tests.py                 # Integrated test runner script
```

---

## B. Backend Component & File Audit

| File Path | Function / Responsibility | Implementation Status |
| :--- | :--- | :--- |
| [`main.py`](file:///d:/Distributed%20RAG/main.py) | Exposes FastAPI endpoints (`/documents`, `/query`, `/health`). | **Real (Production REST API Skeleton)** |
| [`services/document_service.py`](file:///d:/Distributed%20RAG/services/document_service.py) | Coordinates ingestion, indexes chunks, tracks job status. | **Real Logic / Mock Storage** (Uses in-memory `dict`) |
| [`services/query_service.py`](file:///d:/Distributed%20RAG/services/query_service.py) | Security checks, pipeline orchestration, latency measurement. | **Real Logic** (Includes real prompt-injection regex filter & wall-clock timing) |
| [`helpers/document_helper.py`](file:///d:/Distributed%20RAG/helpers/document_helper.py) | Orchestrates text parsing, PII redaction, entities, chunking. | **Simplified Real** (Expects raw text string; Docling PDF parser missing) |
| [`helpers/retrieval_helper.py`](file:///d:/Distributed%20RAG/helpers/retrieval_helper.py) | Executes parallel retrieval, Reciprocal Rank Fusion (RRF), reranking. | **Real Algorithm / Heuristic Reranker** (RRF is real math; Reranker is heuristic word overlap) |
| [`helpers/generation_helper.py`](file:///d:/Distributed%20RAG/helpers/generation_helper.py) | Grounded answer generation. | **Mock / Placeholder** (No LLM call; returns string template of top chunk) |
| [`helpers/citation_helper.py`](file:///d:/Distributed%20RAG/helpers/citation_helper.py) | Builds citation objects and checks chunk text containment. | **Real Logic** (Sub-string / token containment check against chunk text) |
| [`data_access/vector_data_access.py`](file:///d:/Distributed%20RAG/data_access/vector_data_access.py) | Embeddings & vector similarity search. | **Mock / Pseudo-Implementation** (SHA-256 hash projection; in-memory dict cosine search; Pinecone SDK not connected) |
| [`data_access/bm25_data_access.py`](file:///d:/Distributed%20RAG/data_access/bm25_data_access.py) | Inverted index & BM25 exact term retrieval. | **Real Pure-Python BM25 / In-Memory** (Real BM25 TF-IDF math; stored in Python `dict`) |
| [`data_access/graph_data_access.py`](file:///d:/Distributed%20RAG/data_access/graph_data_access.py) | Knowledge Graph & entity relationship traversal. | **Mock / Simplified Graph** (Co-occurrence adjacency dict; Neo4j Bolt driver not connected) |
| [`utils/pii_redactor.py`](file:///d:/Distributed%20RAG/utils/pii_redactor.py) | Redacts SSN, email, phone, API keys, credit cards. | **Real Rule-Based** (Regex-based pattern matching) |
| [`utils/entity_extractor.py`](file:///d:/Distributed%20RAG/utils/entity_extractor.py) | Extracts Org, Contract, Date, Person entities. | **Real Rule-Based** (Regex pattern matching; no NER transformer/spaCy) |
| [`utils/chunker.py`](file:///d:/Distributed%20RAG/utils/chunker.py) | Sliding window paragraph/text chunker with metadata. | **Real Logic** (Character count sliding window) |
| [`eval/evaluator.py`](file:///d:/Distributed%20RAG/eval/evaluator.py) | Calculates Precision@K, Recall@K, MRR, citation validity ratio. | **Real Logic** (Mathematical evaluation formulas) |

---

## C. Specific Audit Responses

1. **Where documents are stored**:
   - Documents are stored **in-memory** inside Python dictionaries (`_DOC_METADATA_STORE` and `_DOC_STATUS_STORE` in `services/document_service.py`). There is **no persistent database** (PostgreSQL, DynamoDB, MongoDB, or S3). All data is lost when the server process restarts.

2. **How chunks are created**:
   - `utils/chunker.py` (`create_semantic_chunks`) splits text by double newlines (`\n\n`) into paragraphs and accumulates them into a sliding window (default `chunk_size = 500` chars, `overlap = 50` chars). It constructs a `ChunkMetadataInterface` containing `document_id`, `chunk_id`, `page`, `section`, `entities`, and `source`.

3. **How retrieval currently works**:
   - `helpers/retrieval_helper.py` (`execute_hybrid_retrieval`) queries three in-memory channels:
     - Vector similarity search (`query_vector_store`)
     - BM25 keyword search (`query_bm25_index`)
     - In-memory entity graph traversal (`query_graph_store`)
   - It fuses the results using Reciprocal Rank Fusion (`apply_rrf_fusion`) with formula $\frac{1}{60 + \text{rank}}$.
   - It re-scores candidates using a heuristic formula (`rrf_score + term_overlap_ratio * 0.2 + multi_source_count * 0.1`).

4. **Whether embeddings are actually generated**:
   - **NO real AI embeddings are generated**. `data_access/vector_data_access.py` uses a deterministic SHA-256 hash projection (`hashlib.sha256(text.encode()).digest()`) to construct a 1536-dimensional float vector. It does not call Bedrock, OpenAI, Cohere, or SentenceTransformers.

5. **Whether Pinecone is actually being used**:
   - **NO**. Pinecone vector DB is not connected. Vector records are stored in an in-memory dictionary `_LOCAL_VECTOR_STORE` and searched using a Python dot-product cosine similarity loop.

6. **Whether BM25 is actually being used**:
   - **YES (Custom In-Memory Implementation)**. `data_access/bm25_data_access.py` implements true BM25 scoring math ($k_1=1.5, b=0.75$), computing TF, IDF, and document length normalization. However, the index lives purely in Python memory (`dict`).

7. **Whether Neo4j is actually being used**:
   - **NO**. Neo4j database is not connected. `data_access/graph_data_access.py` stores entity co-occurrences in Python sets (`_GRAPH_ENTITIES` and `_ENTITY_RELATIONSHIPS`) and performs 1-hop set intersections.

8. **Whether an LLM is actually being called**:
   - **NO LLM IS CALLED**. `helpers/generation_helper.py` takes the top retrieved chunk and formats its raw text into a string template (`f"Based on the provided evidence in {top_meta.source}:\n{top_chunk_text}"`). No AWS Bedrock, OpenAI, Claude, or local LLM is invoked.

9. **Whether reranking is actually implemented**:
   - **PARTIALLY (Heuristic Rule-Based)**. Reranking is performed by custom Python logic calculating term overlap and multi-channel presence. No trained Cross-Encoder reranker model (like Cohere Rerank or BGE-Reranker) is used.

10. **How citations are generated**:
    - `helpers/citation_helper.py` (`build_citations`) extracts `document_id`, `page`, `section`, `chunk_id`, `source`, and a text snippet directly from the `ChunkMetadataInterface` of retrieved evidence chunks.

11. **How citation validity is calculated**:
    - `helpers/citation_helper.py` (`validate_citations`) checks if the citation snippet string exists inside the corresponding chunk's raw text. If substring or word overlap is found, `is_valid` is set to `True`.

12. **How confidence_score is calculated**:
    - `helpers/generation_helper.py` returns a **hardcoded static float `0.95`** whenever evidence is found, and `0.0` if no evidence exists.

13. **How processing_time_ms is calculated**:
    - `services/query_service.py` measures real wall-clock latency using Python's `time.perf_counter()` before and after query execution (`round((end_time - start_time) * 1000.0, 2)`).

---

## D. Why Unrelated Questions Return the Same Chunk

If you ingest a single document (e.g., a contract containing termination notice clauses) and ask unrelated questions like:
- *"What is ABC Corp's annual revenue?"*
- *"What is the termination notice period for Contract #999?"*

The system currently returns the exact same chunk because:
1. **Single Chunk in Store**: If only one document/chunk was ingested into memory, `_LOCAL_VECTOR_STORE` contains only that 1 chunk.
2. **Pseudo-Embeddings Always Match**: SHA-256 hash vectors have non-zero cosine similarity with any query vector. The vector retriever returns the single available chunk as candidate hit #1.
3. **RRF & Reranker Pass It Through**: Even if BM25 and Graph retrievers return 0 hits for "revenue", the vector retriever hit passes through RRF fusion and reranking as the top candidate.
4. **No LLM Guardrail / Relevance Gate**: Because `generation_helper.py` does not use an actual LLM to evaluate question relevance or perform zero-shot evaluation, it blindly takes candidate #1 and wraps it in the answer template.

---

## E. Summary of Mocks, Hardcoded Values & Simplified Logic

1. **Answer Generation**: Mocked (Template string concatenation; no LLM call).
2. **Confidence Score**: Hardcoded (`0.95`).
3. **Embeddings**: Mocked (SHA-256 hash projection; no model embedding call).
4. **Vector Storage**: Mocked (In-memory Python `dict`; no Pinecone API integration).
5. **Graph Database**: Mocked (In-memory co-occurrence dictionary; no Neo4j Cypher connection).
6. **Document Parser**: Simplified (Raw string text input; no Docling PDF layout parser).
7. **Entity Extractor**: Simplified (Regex patterns; no NLP/spacy/transformer model).
8. **Reranker**: Simplified (Heuristic word-overlap formula; no Cross-Encoder model).
9. **Persistence**: Mocked (In-memory storage; no PostgreSQL / DynamoDB / S3).

---

## F. Requirement Comparison & Gap Analysis Matrix

| Requirement | Current Implementation | Real / Mock / Missing | File(s) | What Needs to Change |
| :--- | :--- | :--- | :--- | :--- |
| **1. Docling PDF Ingestion** | Accepts raw text strings via API payload. | **Missing / Mock** | `helpers/document_helper.py`, `main.py` | Integrate `docling` library to parse raw PDF binaries, preserve page layouts, tables, and section structures. |
| **2. PII Detection & Redaction** | Regex patterns for SSN, Email, Phone, API Keys, Credit Cards. | **Real (Rule-Based)** | `utils/pii_redactor.py` | Add Microsoft Presidio / spaCy PII model for contextual PII detection. |
| **3. Entity Extraction** | Regex patterns for Org, Contract, Date, Person. | **Real (Rule-Based)** | `utils/entity_extractor.py` | Integrate spaCy or LLM NER for deep entity extraction. |
| **4. Semantic Chunking** | Sliding window chunker retaining doc/page/section/entities metadata. | **Real** | `utils/chunker.py` | Enhance with layout-aware section header splitting from Docling outputs. |
| **5. Embeddings** | SHA-256 hash vector projection (1536 dim). | **Mock** | `data_access/vector_data_access.py` | Replace SHA-256 hash with AWS Bedrock (`amazon.titan-embed-text-v2`) or OpenAI embeddings (`text-embedding-3-small`). |
| **6. Pinecone Vector Storage** | In-memory `dict` + cosine similarity loop. | **Mock** | `data_access/vector_data_access.py` | Initialize `pinecone-client` SDK, connect to Pinecone index, upsert real vectors, and issue vector queries. |
| **7. BM25 Keyword Retrieval** | Pure-Python BM25 inverted index algorithm. | **Real (In-Memory)** | `data_access/bm25_data_access.py` | Connect to Elasticsearch / OpenSearch or persist inverted index to disk/DB. |
| **8. Neo4j Knowledge Graph** | In-memory co-occurrence adjacency maps. | **Mock** | `data_access/graph_data_access.py` | Connect `neo4j` Python driver, write Cypher `MERGE` statements for nodes/edges, and execute Cypher multi-hop queries. |
| **9. Hybrid Retrieval (Vector + BM25 + Graph)** | Reciprocal Rank Fusion (RRF) combining vector, BM25, and graph. | **Real Algorithm** | `helpers/retrieval_helper.py` | Connect to real vector/graph backends so RRF operates on true multi-source hits. |
| **10. Reranking** | Heuristic term-overlap + multi-source score weighting. | **Simplified / Mock** | `helpers/retrieval_helper.py` | Integrate a neural reranker model (e.g. Cohere Rerank API or `sentence-transformers` Cross-Encoder). |
| **11. LLM Grounded Answer Generation** | String template formatting top chunk text. | **Mock** | `helpers/generation_helper.py` | Call AWS Bedrock (`boto3`) Claude/Titan or OpenAI API with system prompt & retrieved context. |
| **12. Citation System & Validation** | Extracts metadata snippets; checks string containment in chunk. | **Real Logic** | `helpers/citation_helper.py` | Enhance validation using NLI (Natural Language Inference) or LLM self-check for claim entailment. |
| **13. Confidence / Relevance Handling** | Hardcoded float `0.95`. | **Mock** | `helpers/generation_helper.py` | Calculate real confidence based on LLM logprobs, rerank scores, or RAG Triad scores. |
| **14. RAG Golden-Set Evaluation** | Calculates Precision@K, Recall@K, MRR, citation accuracy. | **Real** | `eval/evaluator.py` | Expand golden dataset test cases and integrate automated benchmark regression runs. |
| **15. LangSmith Tracing** | Flag present in config; tracing calls missing. | **Missing** | `services/query_service.py`, `utils/config.py` | Add `@traceable` decorator or LangChain/LangSmith callback handlers to trace pipeline steps. |
| **16. Production FastAPI Service** | FastAPI routes with validation & error handling. | **Real** | `main.py`, `services/` | Add authentication middleware, rate limiting, and async background task processing. |
| **17. Infrastructure (S3 → SQS → ECS → DynamoDB)** | In-memory `dict` storage. | **Missing** | `services/document_service.py` | DevOps Track: Store PDFs in S3, send ingestion jobs to SQS, run worker on ECS Fargate, save metadata in DynamoDB. |
