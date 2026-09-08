/**
 * Centralized API Service Layer
 * All network requests to FastAPI backend pass through this service.
 * Never connects directly to PostgreSQL, Pinecone, Neo4j, or OpenAI.
 */

import {
  DocumentListItem,
  DocumentStatus,
  DocumentMetadata,
  IngestionJob,
  QueryRequest,
  QueryResponse,
  HealthResponse,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers: Record<string, string> = {
      'Accept': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    };

    const config: RequestInit = {
      ...options,
      headers,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        let errorDetail = `Request failed with status ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson.detail) {
            errorDetail = typeof errJson.detail === 'string' 
              ? errJson.detail 
              : JSON.stringify(errJson.detail);
          }
        } catch {
          // If response body is not JSON, use status text
          errorDetail = response.statusText || errorDetail;
        }

        const error = new Error(errorDetail);
        (error as any).status = response.status;
        throw error;
      }

      // If response is 204 No Content
      if (response.status === 204) {
        return {} as T;
      }

      return await response.json() as T;
    } catch (err: any) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        throw new Error('Unable to connect to backend service. Please ensure the API server is running.');
      }
      throw err;
    }
  }

  /**
   * Health check endpoint
   */
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  /**
   * List all documents with status and ingestion progress
   */
  async listDocuments(): Promise<DocumentListItem[]> {
    return this.request<DocumentListItem[]>('/documents');
  }

  /**
   * Asynchronously upload a PDF document (returns 202 Accepted + job_id)
   */
  async uploadDocument(file: File, title?: string): Promise<DocumentStatus> {
    const formData = new FormData();
    formData.append('file', file);
    if (title && title.trim()) {
      formData.append('title', title.trim());
    }

    const url = `${this.baseUrl}/documents`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      let errorDetail = `Upload failed with status ${response.status}`;
      try {
        const errJson = await response.json();
        if (errJson.detail) {
          errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch {
        errorDetail = response.statusText || errorDetail;
      }
      const error = new Error(errorDetail);
      (error as any).status = response.status;
      throw error;
    }

    return await response.json() as DocumentStatus;
  }

  /**
   * Retrieve real-time document status & stage progress
   */
  async getDocumentStatus(documentId: string): Promise<DocumentStatus> {
    return this.request<DocumentStatus>(`/documents/${encodeURIComponent(documentId)}/status`);
  }

  /**
   * Retrieve document metadata
   */
  async getDocumentMetadata(documentId: string): Promise<DocumentMetadata> {
    return this.request<DocumentMetadata>(`/documents/${encodeURIComponent(documentId)}`);
  }

  /**
   * Retrieve ingestion job details
   */
  async getJobDetails(jobId: string): Promise<IngestionJob> {
    return this.request<IngestionJob>(`/jobs/${encodeURIComponent(jobId)}`);
  }

  /**
   * Cooperatively cancel document ingestion
   */
  async cancelDocument(documentId: string): Promise<{ status: string; document_id: string }> {
    return this.request<{ status: string; document_id: string }>(
      `/documents/${encodeURIComponent(documentId)}/cancel`,
      { method: 'POST' }
    );
  }

  /**
   * Delete document and all document-owned chunks, vectors, graph nodes, and BM25 entries
   */
  async deleteDocument(documentId: string): Promise<{ status: string; document_id: string }> {
    return this.request<{ status: string; document_id: string }>(
      `/documents/${encodeURIComponent(documentId)}`,
      { method: 'DELETE' }
    );
  }


  /**
   * Execute Hybrid RAG Knowledge Search
   */
  async searchKnowledge(request: QueryRequest): Promise<QueryResponse> {
    return this.request<QueryResponse>('/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
  }
}

export const api = new ApiClient();
export default api;
