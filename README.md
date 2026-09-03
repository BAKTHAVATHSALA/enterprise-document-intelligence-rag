# Enterprise Document Intelligence & Hybrid RAG Platform — Phase 1

Production-grade AI platform for layout-aware PDF document ingestion using **Docling**, PII protection, entity extraction, semantic layout chunking, multi-modal retrieval, grounded answer generation, citation validation, FastAPI REST APIs, and automated evaluation metrics.

---

## 🏗️ Phase 1 Architecture & Flow

Phase 1 establishes a production-quality, real PDF document ingestion foundation:

```text
PDF Upload (Multipart)
        │
        ▼
   FastAPI Endpoint (`POST /documents`)
        │
        ▼
   Docling PDF Parser (`utils/pdf_parser.py`)
        │
        ▼
Structured Document (Pages, Headings, Paragraphs, Tables)
        │
        ▼
 PII Redaction (`utils/pii_redactor.py`)
        │
        ▼
 Entity Extraction (`utils/entity_extractor.py`)
        │
        ▼
 Layout-Aware Semantic Chunking (`utils/chunker.py`)
        │
        ▼
  Chunk Metadata Lineage (Document ID, Page, Section, Entities, Source)
        │
        ▼
    Phase 1 Result
```

---

## 🛠️ Technology Stack & Dependencies

- **Language**: Python 3.13
- **PDF Layout Parser**: `Docling` (`docling` & `docling-core`) / `pypdf`
- **API Framework**: FastAPI + Uvicorn + `python-multipart` + Pydantic v2
- **PDF Fixtures Generator**: `reportlab`
- **Testing Suite**: Pytest 9.1+

---

## 📋 What is Implemented in Phase 1 vs Future Phases

### ✅ Implemented in Phase 1 (Real Pipeline)
1. **Real PDF File Ingestion**: Accepts multipart `.pdf` file uploads. Validates extension, magic headers (`%PDF-`), file sizes, and rejects corrupted files.
2. **Docling PDF Layout Parsing**: Extracts structured document layout elements (pages, headings, paragraphs, and tables).
3. **PII Protection**: Masking of SSNs (`[REDACTED_SSN]`), Emails (`[REDACTED_EMAIL]`), Phone Numbers (`[REDACTED_PHONE]`), API Keys, and Credit Cards before downstream processing.
4. **Entity Extraction**: Identifies Organizations, Contracts, Dates, People, and Locations.
5. **Layout-Aware Semantic Chunker**: Generates chunks preserving page numbers, section headers, document lineage, and entity lists in `ChunkMetadataInterface`.
6. **Real Test PDF Fixtures**: Includes 3 multi-page PDF files in `test_documents/` (`contract_abc.pdf`, `contract_xyz.pdf`, `technical_policy.pdf`).
7. **Automated Test Suite**: 13 automated unit & integration test cases.

### ⏳ Intentionally Reserved for Future Phases
- Pinecone Cloud Vector Index
- AWS Bedrock / OpenAI Real AI Embeddings & LLM Generation
- Neo4j Cypher Database Connection
- Neural Cross-Encoder Reranker (Cohere / BGE)
- LangSmith Tracing Callbacks
- DevOps Infrastructure (AWS S3, SQS, ECS Fargate, DynamoDB)

---

## 📂 Repository File Tree

```text
d:\Distributed RAG
├── test_documents/              # REAL Test PDF Fixtures
│   ├── contract_abc.pdf         # 2-page PDF (ABC Corp, Contract #123, 30 days notice, PII)
│   ├── contract_xyz.pdf         # 2-page PDF (XYZ Technologies, Contract #456, 60 days notice)
│   └── technical_policy.pdf     # 2-page PDF (Headings, policies, structured data table)
├── data_access/
│   ├── bm25_data_access.py      # BM25 keyword inverted index
│   ├── graph_data_access.py     # In-memory entity relationship graph
│   └── vector_data_access.py    # Vector store embeddings & similarity search
├── eval/
│   └── evaluator.py             # Precision@K, Recall@K, MRR & latency evaluator
├── helpers/
│   ├── citation_helper.py       # Citation extraction & claim validation engine
│   ├── document_helper.py       # Docling PDF ingestion & parsing orchestrator
│   ├── generation_helper.py     # Grounded answer synthesis engine
│   └── retrieval_helper.py      # Hybrid multi-retriever search & RRF fusion
├── interfaces/
│   ├── config_interface.py      # Application config schema
│   ├── document_interface.py    # Document, Chunk, PII, Entity schemas
│   ├── query_interface.py       # Query request/response & citation schemas
│   └── retrieval_interface.py   # Retrieval candidate & RRF schemas
├── services/
│   ├── document_service.py      # PDF ingestion job management & status tracking
│   └── query_service.py         # Query execution & prompt injection defense
├── tests/
│   └── test_pipeline.py         # Automated pytest test suite (13 test cases)
├── utils/
│   ├── chunker.py               # Layout-aware semantic chunking utility
│   ├── config.py                # Environment configuration loader
│   ├── entity_extractor.py      # Named entity extraction utility
│   ├── logger.py                # Structured logger utility ([funcName] - message)
│   ├── pdf_parser.py            # Docling PDF layout parser module
│   └── pii_redactor.py          # PII detection & masking utility
├── create_test_pdfs.py          # Script to generate PDF test fixtures
├── main.py                      # FastAPI REST application entry point
├── README.md                    # System documentation
├── rules.md                     # Coding rules & standards specification
└── run_tests.py                 # Integrated test runner script
```

---

## ⚡ Setup & Execution Instructions

### 1. Install Dependencies
```bash
pip install docling fastapi uvicorn pydantic python-multipart reportlab pytest
```

### 2. Generate PDF Test Fixtures (Optional)
```bash
python create_test_pdfs.py
```

### 3. Run FastAPI Application Server
```bash
uvicorn main:app --reload --port 8000
```
Interactive Swagger UI documentation is available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing Swagger PDF Upload

1. Open [http://localhost:8000/docs](http://localhost:8000/docs).
2. Expand `POST /documents`.
3. Click **Try it out**.
4. Choose a PDF file (e.g. `test_documents/contract_abc.pdf`).
5. Click **Execute**.

### Example Ingestion Output (HTTP 201 Created):
```json
{
  "document_id": "doc_063b86f5ae53",
  "status": "COMPLETED",
  "error_message": null,
  "processed_chunks": 2
}
```

---

## 🧪 Running Automated Tests

Run full test suite via Pytest:
```bash
python -m pytest -v
```

Run test suite and golden-set benchmark evaluation:
```bash
python run_tests.py
```
