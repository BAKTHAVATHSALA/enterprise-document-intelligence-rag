/**
 * API Type Definitions for Enterprise Document Intelligence RAG Frontend
 * Strictly aligned with FastAPI Pydantic domain models.
 */

export type DocumentStatusEnum = 
  | 'PENDING'
  | 'PROCESSING'
  | 'COMPLETED'
  | 'PARTIAL'
  | 'FAILED'
  | 'CANCELLED';

export type IngestionStageEnum = 
  | 'QUEUED'
  | 'PARSING'
  | 'PII_REDACTION'
  | 'ENTITY_EXTRACTION'
  | 'CHUNKING'
  | 'EMBEDDING'
  | 'INDEXING'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED';

export type CitationValidationStatusEnum = 
  | 'VALID'
  | 'INVALID'
  | 'FABRICATED'
  | 'MISSING';

export interface DocumentListItem {
  document_id: string;
  tenant_id: string;
  title: string;
  source_filename: string;
  total_pages: number;
  total_chunks: number;
  status: DocumentStatusEnum;
  stage?: IngestionStageEnum;
  progress_percent?: number;
  error_message?: string | null;
  created_at: string;
  job_id?: string | null;
}

export interface DocumentStatus {
  job_id?: string | null;
  document_id: string;
  tenant_id?: string;
  status: DocumentStatusEnum;
  stage?: IngestionStageEnum | null;
  progress_percent?: number;
  error_message?: string | null;
  processed_chunks: number;
  correlation_id?: string | null;
}

export interface DocumentMetadata {
  document_id: string;
  tenant_id: string;
  title: string;
  source_filename: string;
  total_pages: number;
  total_chunks: number;
  created_at: string;
}

export interface IngestionJob {
  job_id: string;
  document_id: string;
  tenant_id: string;
  status: DocumentStatusEnum;
  stage: IngestionStageEnum;
  progress_percent: number;
  retry_count: number;
  max_retries: number;
  error_message?: string | null;
  correlation_id?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface Citation {
  document_id: string;
  page: number;
  section: string;
  chunk_id?: string;
  source?: string;
  snippet?: string;
  is_valid: boolean;
  validation_status: CitationValidationStatusEnum;
  validation_message?: string;
}

export interface CitationValidationSummary {
  total_citations: number;
  valid_count: number;
  invalid_count: number;
  fabricated_count: number;
  missing_count: number;
  is_fully_validated: boolean;
}

export interface QueryRequest {
  question: string;
  document_ids?: string[] | null;
  top_k?: number;
}

export interface QueryResponse {
  question: string;
  answer: string;
  citations: Citation[];
  validation_summary?: CitationValidationSummary | null;
  confidence_score: number;
  processing_time_ms: number;
}

export interface HealthResponse {
  status: string;
  service: string;
}

export interface ApiError {
  detail: string;
  error_type?: string;
  correlation_id?: string;
  status_code?: number;
}
