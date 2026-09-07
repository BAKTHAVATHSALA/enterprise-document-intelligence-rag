# Enterprise Document Intelligence & Hybrid RAG Platform

Production-grade AI platform for layout-aware PDF document ingestion using **Docling**, PII protection, entity extraction, semantic layout chunking, multi-modal hybrid retrieval (Pinecone Vector + Inverted BM25 + Neo4j Graph RAG), neural cross-encoder reranking, grounded answer generation with citation verification, PostgreSQL primary persistence, structured JSON logging, and **Phase 6 Asynchronous Ingestion Architecture**.

---

## 🏗️ System Architecture & Ingestion Flow

```mermaid
flowchart TD
    Client([Client / API Consumer])
    
    subgraph FastAPI["FastAPI Application Layer (main.py)"]
        Upload["POST /documents<br/>(Validates file, generates doc_id + job_id, returns 202)"]
        DocStatus["GET /documents/{id}/status"]
        JobStatus["GET /jobs/{job_id}"]
        CancelDoc["POST /documents/{id}/cancel"]
        CancelJob["POST /jobs/{job_id}/cancel"]
        Query["POST /query<br/>(Prompt injection defense, hybrid retrieval, synthesis)"]
    end

    subgraph QueueLayer["Queue Abstraction Layer (services/queue_service.py)"]
        QueueIF["JobQueueInterface"]
        MemoryQueue["AsyncInMemoryJobQueue<br/>(Loop-independent queue + cancellation tracking)"]
    end

    subgraph WorkerLayer["Worker Process Layer (workers/ingestion_worker.py)"]
        Worker["IngestionWorker Loop<br/>(Dequeues, checks cancellation, handles retries)"]
        
        subgraph Stages["7-Stage Pipeline"]
            S1["1. PARSING (15% - Docling)"]
            S2["2. PII_REDACTION (30%)"]
            S3["3. ENTITY_EXTRACTION (45%)"]
            S4["4. CHUNKING (60%)"]
            S5["5. EMBEDDING (75% - OpenAI)"]
            S6["6. INDEXING (90% - Derived Stores)"]
            S7["7. COMPLETED (100%)"]
        end
    end

    subgraph Storage["Storage & Persistence"]
        PG[("PostgreSQL<br/>(documents, document_chunks, ingestion_jobs)") ]
        Pinecone[("Pinecone Vector DB<br/>(Derived dense vectors)")]
        Neo4j[("Neo4j Graph Store<br/>(Derived entity graph)")]
        BM25["In-Memory BM25 Index<br/>(Hydrated from PG, deduplicated)"]
    end

    Client -->|1. Upload PDF| Upload
    Upload -->|2. Create DB Records (PENDING/QUEUED)| PG
    Upload -->|3. Enqueue Payload| MemoryQueue
    Upload -->|4. Return 202 Accepted + job_id| Client
    
    Client -->|Poll Status| DocStatus
    Client -->|Poll Job| JobStatus
    DocStatus -->|Read Status & Stage| PG
    JobStatus -->|Read Job Record| PG
    
    Client -->|Cancel Job| CancelDoc
    CancelDoc -->|Mark Cancelled| MemoryQueue
    CancelDoc -->|Update Status to CANCELLED| PG

    Worker -->|Dequeue Job| MemoryQueue
    Worker -->|Execute Pipeline| S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    Worker -->|Persist Metadata & Chunks FIRST| PG
    S6 -->|Index Vectors| Pinecone
    S6 -->|Index Entities| Neo4j
    S6 -->|Index Chunks| BM25
    Worker -->|Update Stage & Progress %| PG
```

---

## 🚀 Phase 6: Asynchronous Ingestion Architecture

### 1. Consistent Asynchronous Contract (`POST /documents`)
- `POST /documents` is **consistently asynchronous** across all environments and clients.
- Synchronously validates file extension (`.pdf`), empty payloads, and magic header bytes (`%PDF-`).
- Immediately generates `document_id` and `job_id`, records them in PostgreSQL with `status="PENDING"` and `stage="QUEUED"`, enqueues the job, and returns **`HTTP 202 Accepted`**.

#### Response Schema (`HTTP 202 Accepted`):
```json
{
  "document_id": "doc_063b86f5ae53",
  "job_id": "job_e9f1a23b8c44",
  "status": "PENDING",
  "stage": "QUEUED",
  "progress_percent": 0.0,
  "processed_chunks": 0,
  "error_message": null,
  "correlation_id": "corr_7a9f2b1c8e"
}
```

### 2. Extensible Queue Abstraction (`JobQueueInterface`)
- Backed by an `AsyncInMemoryJobQueue` using a thread-safe, loop-independent deque with cooperative polling.
- Enables swapping with Redis or cloud message brokers in future iterations without modifying service or worker code.

### 3. Pipeline Stages & Real Progress Tracking
Worker reports granular progress percentages and updates `ingestion_jobs` and `documents` tables in PostgreSQL:
1. **`PARSING` (15%)**: Real PDF document layout extraction using Docling.
2. **`PII_REDACTION` (30%)**: Redaction of SSNs, Emails, Phone Numbers, API keys, and Credit Cards.
3. **`ENTITY_EXTRACTION` (45%)**: Extraction of Organizations, Contracts, Dates, People, Locations.
4. **`CHUNKING` (60%)**: Layout-aware semantic chunking preserving document lineage.
5. **`EMBEDDING` (75%)**: Batch dense vector generation via OpenAI embeddings.
6. **`INDEXING` (90%)**:
   - Primary persistence to PostgreSQL (`documents` and `document_chunks`).
   - Derived indexing to Pinecone, Neo4j, and BM25.
   - Bounded exponential retries (up to 3 attempts: `0.25s`, `0.5s`, `1.0s`, `2.0s`) for transient failures.
7. **`COMPLETED` (100%)**: Finalized and ready for retrieval.

### 4. Cooperative Job Cancellation
- Clients can cancel running or queued jobs via `POST /documents/{id}/cancel` or `POST /jobs/{job_id}/cancel`.
- Worker checks cancellation state before and between each stage; aborts gracefully, updates status to `CANCELLED`, and skips derived indexing.

### 5. BM25 Idempotency & Frequency Deduplication
- Re-ingesting a document evicts previous document frequency contributions from `_BM25_DF` before re-indexing, preventing term frequency inflation.

---

## 🛠️ Technology Stack & Dependencies

- **Language**: Python 3.13
- **PDF Layout Parser**: `Docling` (`docling` & `docling-core`) / `pypdf`
- **Primary Relational Store**: PostgreSQL (Neon / Local) with `psycopg2-binary`
- **Vector Store**: Pinecone Serverless Vector Index
- **Graph Database**: Neo4j Graph Database
- **Keyword Search**: In-Memory BM25 with dynamic inverted indexing & persistence hydration
- **Reranker**: Cross-Encoder Neural Reranker (`sentence-transformers`)
- **LLM Synthesis**: OpenAI GPT-4o / GPT-4o-mini
- **API Framework**: FastAPI + Uvicorn + Pydantic v2
- **Tracing & Evaluation**: LangSmith Tracing + RAGAS 0.4.3 Benchmark Suite
- **Testing Suite**: Pytest 9.1+

---

## ⚡ Setup & Execution Instructions

### 1. Environment Configuration
Ensure `.env` contains the required credentials:
```ini
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=rag-eval
NEO4J_URI=neo4j+s://...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
POSTGRES_DB_URL=postgresql://...
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=distributed-rag
```

### 2. Run Application Server
```bash
uvicorn main:app --reload --port 8000
```
Interactive Swagger UI documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 API Usage

### Ingest a Document (Async 202):
```bash
curl -X POST "http://localhost:8000/documents" \
  -H "accept: application/json" \
  -F "file=@test_documents/contract_abc.pdf"
```

### Check Ingestion Status:
```bash
curl -X GET "http://localhost:8000/documents/{document_id}/status"
```

### Check Job Details:
```bash
curl -X GET "http://localhost:8000/jobs/{job_id}"
```

### Cancel Ingestion:
```bash
curl -X POST "http://localhost:8000/documents/{document_id}/cancel"
```

### Query Documents:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the termination notice period for ABC Corp?"}'
```

---

## 🧪 Automated Testing

### Run Phase 6 Async Ingestion Tests:
```bash
python -m pytest tests/test_phase6_async.py -v
```

### Run Regression & Full Test Suite:
```bash
python -m pytest tests/ -v
```
