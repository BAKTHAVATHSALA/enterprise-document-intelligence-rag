CAPSTONE 01 — PART 1
Backend & AI Application Engineering
Enterprise Document Intelligence & Hybrid RAG Platform • Hackathon Project Specification
1. Project Overview
One-line description: A production-grade AI platform that ingests enterprise documents, understands their structure and
entities, performs hybrid retrieval across semantic, keyword and graph sources, and generates evidence-grounded
answers with verifiable citations.
Backend / AI ownership: This track owns the complete AI and application layer—from document understanding
and RAG to APIs, evaluation, security logic and the user-facing application contract.
2. Problem Statement
Enterprises store thousands of PDFs and business documents containing critical information across contracts, policies,
reports, manuals, invoices and technical documentation. Traditional keyword search struggles with semantic meaning,
while basic RAG systems can fail when exact identifiers matter, information is distributed across documents, or
relationships between entities must be understood.
Challenge: Build an AI-powered document intelligence platform that can process documents, extract useful
knowledge, retrieve the strongest evidence using multiple strategies, and answer questions with traceable
citations.
3. Proposed AI Solution
The AI/application layer combines layout-aware document processing, PII redaction, entity extraction, semantic chunking,
embeddings, vector search, BM25 keyword search, Neo4j knowledge-graph retrieval, result fusion, reranking, LLM
generation and citation validation.
4. Functional Capabilities
Capability
Backend / AI Responsibility
Document ingestion
Document understanding
PII protection
Entity extraction
Semantic chunking
Embeddings
Vector retrieval
Keyword retrieval
Accept document jobs and process PDFs through the ingestion pipeline.
Use Docling to preserve layout, pages, sections, paragraphs and tables.
Detect and redact sensitive information before downstream AI processing.
Identify people, organizations, contracts, locations, dates and other relevant entities.
Create meaningful chunks with document/page/section/entity metadata.
Convert chunks into semantic vectors using the selected embedding model.
Query Pinecone for semantically similar evidence.
Use BM25 for exact terms, names, IDs and identifiers.
Graph retrieval
Hybrid fusion
Reranking
Grounded generation
Citation system
Citation validation
Evaluation
Use Neo4j/Cypher for entity relationships and multi-hop retrieval.
Combine vector, keyword and graph candidates.
Re-score candidates and select the strongest evidence.
Generate answers using retrieved evidence and controlled prompts.
Return document, page and chunk references supporting claims.
Check that cited evidence actually supports the generated answer.
Run golden-set tests and measure retrieval/generation quality.
5. AI / RAG Pipeline
Ingestion: PDF fi Docling fi structured document fi PII redaction fi entity extraction fi semantic chunking fi
metadata fi embeddings fi Pinecone + Neo4j.
Query: User question fi Vector Search + BM25 + Graph Search fi fusion fi reranking fi context selection fi LLM fi
answer + validated citations.
6. Data Model / Metadata
Every chunk should retain enough metadata to trace it back to its origin.
Field
Example
document_id
chunk_id
page
section
text
doc_8f92a71
chunk_982
17
Termination
The agreement may be terminated...
entities
source
7. Retrieval Strategy
ABC Corp, Contract 123
contract.pdf
Vector: semantic meaning. BM25: exact words, names and identifiers. Graph: relationships and multi-hop questions.
The three result sets are fused and passed to a reranker before generation.
Example: a question about contracts with ABC Corp can require semantic matching for the intent, exact matching for a
contract identifier, and graph traversal to connect ABC Corp fi contract fi product fi supplier.
8. Generation & Citations
The LLM receives the user question plus selected evidence. It must avoid unsupported claims, state when evidence is
insufficient, distinguish facts from inference, and attach source references.
Example response: “The agreement requires 30 days written notice.” Source: Contract_ABC.pdf, Page 17.
9. API Contract
Endpoint Purpose
POST /documents Create a document ingestion job.
GET /documents/{document_id} Return document metadata.
GET /documents/{document_id}/status Return processing status.
POST /query Answer a natural-language question using the RAG pipeline.
Example query payload
{"question":"What is the termination period?"}
Example response contains the answer plus citation objects containing document, page and chunk identifiers.
10. Evaluation
Area Measures
Retrieval Precision@K, Recall@K, MRR
Generation Faithfulness, context relevance, answer relevance
RAG RAG Triad
Citations Citation correctness / evidence support
Performance Latency, throughput, processing time
Cost Input/output tokens, embedding calls, LLM calls, cost/query
Reliability Retries, failure handling and regression tests
11. Observability
Trace the complete request path: Query fi Retriever fi Pinecone/BM25/Neo4j fi Fusion fi Reranker fi LLM fi Answer.
Capture latency, token usage, retrieved chunks, prompts, responses, failures and evaluation scores using LangSmith
where appropriate.
12. Backend Security & Reliability
• Treat document content as untrusted data; defend against prompt injection.
• Enforce document-level authorization so users cannot retrieve documents they should not see.
• Validate query size, retrieved context and model output limits.
• Handle invalid/corrupt PDFs, parser failures, embedding failures, database failures, LLM timeouts and rate limits.
• Implement retries where safe and return meaningful API errors.
• Keep secrets out of source control; use environment/secret management.
• Consider caching repeated queries and controlling token budgets to reduce cost.
13. Technology Stack — Backend / AI
Category Technology Purpose
Language Python AI pipeline and backend
API FastAPI REST service
Document AI Docling Layout-aware parsing
LLM AWS Bedrock Answer generation
Category
Embeddings
Vector DB
Keyword
Graph DB
Technology
AWS Bedrock Embeddings
Pinecone
BM25
Neo4j + Cypher
Purpose
Semantic vectors
Vector retrieval
Exact-term retrieval
Relationships / multi-hop retrieval
Orchestration
Observability
Testing
Frontend contract
Source control
LangChain / LangGraph
LangSmith
Pytest
Next.js / React
Git / GitHub
14. Backend Deliverable
Reusable AI workflows where useful
Tracing and evaluation
Automated tests
Web application integration
Collaboration
A tested AI backend that can accept document jobs, process and index documents, answer questions through
hybrid RAG, return source-backed citations, expose clean APIs, produce traces/evaluation results, and provide
stable interfaces for the DevOps/infrastructure track