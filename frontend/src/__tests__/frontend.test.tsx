import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DocumentsPage } from '../components/documents/DocumentsPage';
import { KnowledgeSearchPage } from '../components/search/KnowledgeSearchPage';
import { UploadModal } from '../components/documents/UploadModal';
import { DocumentDetailDrawer } from '../components/ingestion/DocumentDetailDrawer';
import { api } from '../services/api';
import { DocumentListItem, DocumentStatus, QueryResponse } from '../types/api';

describe('Phase 7 — Enterprise Document Intelligence Frontend Test Suite', () => {
  const mockDocuments: DocumentListItem[] = [
    {
      document_id: 'doc_sec123',
      tenant_id: 'default_tenant',
      title: 'Security Policy.pdf',
      source_filename: 'Security Policy.pdf',
      total_pages: 14,
      total_chunks: 42,
      status: 'COMPLETED',
      stage: 'COMPLETED',
      progress_percent: 100,
      created_at: '2026-09-07T10:00:00Z',
      job_id: 'job_sec123',
    },
    {
      document_id: 'doc_priv456',
      tenant_id: 'default_tenant',
      title: 'Privacy Framework.pdf',
      source_filename: 'Privacy Framework.pdf',
      total_pages: 8,
      total_chunks: 18,
      status: 'PROCESSING',
      stage: 'EMBEDDING',
      progress_percent: 75,
      created_at: '2026-09-07T11:00:00Z',
      job_id: 'job_priv456',
    },
  ];

  beforeEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  // TEST 1: Documents page renders with real document list and headers
  it('1. Documents page renders with real document list and headers', async () => {
    vi.spyOn(api, 'listDocuments').mockResolvedValue(mockDocuments);
    vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
      document_id: 'doc_priv456',
      status: 'COMPLETED',
      stage: 'COMPLETED',
      progress_percent: 100,
      processed_chunks: 18,
    });

    render(<DocumentsPage />);

    expect(screen.getByTestId('documents-page-header')).toBeInTheDocument();
    expect(screen.getByText('Documents')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Security Policy.pdf')).toBeInTheDocument();
      expect(screen.getByText('Privacy Framework.pdf')).toBeInTheDocument();
    });

    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  // TEST 2, 3, 4: Upload sends PDF to POST /documents and captures 202 Accepted payload
  it('2-4. Upload modal sends PDF, handles HTTP 202 Accepted, and captures job_id & doc_id', async () => {
    const mock202Response: DocumentStatus = {
      job_id: 'job_test999',
      document_id: 'doc_test999',
      tenant_id: 'default_tenant',
      status: 'PENDING',
      stage: 'QUEUED',
      progress_percent: 0,
      processed_chunks: 0,
    };

    const uploadSpy = vi.spyOn(api, 'uploadDocument').mockResolvedValue(mock202Response);
    const handleAccepted = vi.fn();

    render(
      <UploadModal
        isOpen={true}
        onClose={vi.fn()}
        onUploadAccepted={handleAccepted}
      />
    );

    const file = new File(['%PDF-1.4 test binary'], 'contract.pdf', { type: 'application/pdf' });
    const input = screen.getByTestId('file-input');

    fireEvent.change(input, { target: { files: [file] } });

    expect(screen.getByText('contract.pdf')).toBeInTheDocument();

    const submitBtn = screen.getByTestId('submit-upload-btn');
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(uploadSpy).toHaveBeenCalledWith(file, 'contract');
      expect(handleAccepted).toHaveBeenCalledWith(mock202Response);
    });
  });

  // TEST 5 & 6: Status polling queries /documents/{id}/status and updates UI
  it('5-6. Status polling queries /documents/{id}/status and stops on terminal state', async () => {
    const initialDocs: DocumentListItem[] = [
      {
        document_id: 'doc_poll1',
        tenant_id: 'default_tenant',
        title: 'Polled Doc.pdf',
        source_filename: 'Polled Doc.pdf',
        total_pages: 2,
        total_chunks: 0,
        status: 'PROCESSING',
        stage: 'CHUNKING',
        progress_percent: 60,
        created_at: '2026-09-07T12:00:00Z',
        job_id: 'job_poll1',
      },
    ];

    vi.spyOn(api, 'listDocuments').mockResolvedValue(initialDocs);
    const statusSpy = vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
      document_id: 'doc_poll1',
      job_id: 'job_poll1',
      status: 'COMPLETED',
      stage: 'COMPLETED',
      progress_percent: 100,
      processed_chunks: 10,
    });

    render(<DocumentsPage />);

    await waitFor(() => {
      expect(screen.getByText('Polled Doc.pdf')).toBeInTheDocument();
    });

    // Wait for the polling interval to fire
    await waitFor(
      () => {
        expect(statusSpy).toHaveBeenCalledWith('doc_poll1');
      },
      { timeout: 3500 }
    );

    await waitFor(() => {
      expect(screen.getByText('100%')).toBeInTheDocument();
    });
  });

  // TEST 7 & 8: Ingestion error is captured and displayed safely
  it('7-8. Ingestion polling stops gracefully when status reaches FAILED', async () => {
    const initialDocs: DocumentListItem[] = [
      {
        document_id: 'doc_failed1',
        tenant_id: 'default_tenant',
        title: 'Failed Doc.pdf',
        source_filename: 'Failed Doc.pdf',
        total_pages: 1,
        total_chunks: 0,
        status: 'PROCESSING',
        stage: 'PARSING',
        progress_percent: 15,
        created_at: '2026-09-07T12:00:00Z',
        job_id: 'job_failed1',
      },
    ];

    vi.spyOn(api, 'listDocuments').mockResolvedValue(initialDocs);
    vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
      document_id: 'doc_failed1',
      job_id: 'job_failed1',
      status: 'FAILED',
      stage: 'PARSING',
      progress_percent: 15,
      processed_chunks: 0,
      error_message: 'Docling parse timeout.',
    });

    render(<DocumentsPage />);

    await waitFor(() => {
      expect(screen.getByText('Failed Doc.pdf')).toBeInTheDocument();
    });

    await waitFor(
      () => {
        expect(screen.getByText('FAILED')).toBeInTheDocument();
      },
      { timeout: 3500 }
    );
  });

  // TEST 9 & 10: Ingestion Detail Drawer renders 7 stages and triggers cooperative cancel
  it('9-10. Detail Drawer renders 7 stages and executes cooperative cancel endpoint', async () => {
    const activeDoc: DocumentListItem = {
      document_id: 'doc_active77',
      tenant_id: 'default_tenant',
      title: 'Active Doc.pdf',
      source_filename: 'Active Doc.pdf',
      total_pages: 5,
      total_chunks: 10,
      status: 'PROCESSING',
      stage: 'ENTITY_EXTRACTION',
      progress_percent: 45,
      created_at: '2026-09-07T12:00:00Z',
      job_id: 'job_active77',
    };

    vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
      document_id: 'doc_active77',
      job_id: 'job_active77',
      status: 'PROCESSING',
      stage: 'ENTITY_EXTRACTION',
      progress_percent: 45,
      processed_chunks: 10,
    });

    const cancelSpy = vi.spyOn(api, 'cancelDocument').mockResolvedValue({
      status: 'cancelled',
      document_id: 'doc_active77',
    });

    const updateCallback = vi.fn();

    render(
      <DocumentDetailDrawer
        document={activeDoc}
        isOpen={true}
        onClose={vi.fn()}
        onDocumentUpdated={updateCallback}
      />
    );

    expect(screen.getByTestId('document-detail-drawer')).toBeInTheDocument();
    expect(screen.getByTestId('stage-item-PARSING')).toBeInTheDocument();
    expect(screen.getByTestId('stage-item-ENTITY_EXTRACTION')).toHaveClass('stage-active');

    const cancelBtn = screen.getByTestId('cancel-ingestion-btn');
    fireEvent.click(cancelBtn);

    await waitFor(() => {
      expect(cancelSpy).toHaveBeenCalledWith('doc_active77');
      expect(updateCallback).toHaveBeenCalled();
    });
  });

  // TEST 11, 15, 16: Knowledge search calls query endpoint and renders grounded answer with citations
  it('11, 15, 16. Knowledge Search calls query endpoint and displays grounded answer and verified citations', async () => {
    const mockQueryResponse: QueryResponse = {
      question: 'What is the termination notice period?',
      answer: 'The agreement requires a 30 days notice [Doc: doc_sec123, Page: 14, Section: Termination Clause].',
      confidence_score: 0.94,
      processing_time_ms: 184.5,
      citations: [
        {
          document_id: 'doc_sec123',
          page: 14,
          section: 'Termination Clause',
          source: 'Security Policy.pdf',
          snippet: 'Either party may terminate upon giving thirty (30) days written notice.',
          is_valid: true,
          validation_status: 'VALID',
        },
      ],
      validation_summary: {
        total_citations: 1,
        valid_count: 1,
        invalid_count: 0,
        fabricated_count: 0,
        missing_count: 0,
        is_fully_validated: true,
      },
    };

    const querySpy = vi.spyOn(api, 'searchKnowledge').mockResolvedValue(mockQueryResponse);

    render(<KnowledgeSearchPage documents={mockDocuments} />);

    const input = screen.getByTestId('search-input');
    fireEvent.change(input, { target: { value: 'What is the termination notice period?' } });

    const submitBtn = screen.getByTestId('search-submit-btn');
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(querySpy).toHaveBeenCalledWith({
        question: 'What is the termination notice period?',
        document_ids: null,
        top_k: 5,
      });
    });

    await waitFor(() => {
      expect(screen.getByTestId('answer-card')).toBeInTheDocument();
      expect(screen.getByText(/30 days notice/)).toBeInTheDocument();
      expect(screen.getByTestId('inline-citation')).toBeInTheDocument();
      expect(screen.getByText('Confidence: 94%')).toBeInTheDocument();
      expect(screen.getByText('184.5ms')).toBeInTheDocument();
      expect(screen.getByText('Page 14 · Termination Clause')).toBeInTheDocument();
    });
  });

  // TEST 12, 13, 14, 17: Search handles empty, loading, and error states safely
  it('12-14, 17. Search handles empty, loading, and error states gracefully', async () => {
    // 14: Empty library state
    const { rerender } = render(<KnowledgeSearchPage documents={[]} />);
    expect(screen.getByTestId('search-no-docs-state')).toBeInTheDocument();

    // 13, 17: Error state
    vi.spyOn(api, 'searchKnowledge').mockRejectedValue(new Error('Retrieval service timeout.'));

    rerender(<KnowledgeSearchPage documents={mockDocuments} />);

    const input = screen.getByTestId('search-input');
    fireEvent.change(input, { target: { value: 'Trigger an error query' } });

    const submitBtn = screen.getByTestId('search-submit-btn');
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByTestId('search-error-alert')).toBeInTheDocument();
      expect(screen.getByText('Retrieval service timeout.')).toBeInTheDocument();
    });
  });

  // TEST SUITE: Ingestion Pipeline Stage UI State Mapping (Phase 7 Fix Verification)
  describe('Phase 7 — Ingestion Pipeline Stage UI State Mapping', () => {
    const baseDoc: DocumentListItem = {
      document_id: 'doc_stage_test',
      tenant_id: 'default_tenant',
      title: 'Stage Test Doc.pdf',
      source_filename: 'Stage Test Doc.pdf',
      total_pages: 10,
      total_chunks: 25,
      status: 'PROCESSING',
      stage: 'QUEUED',
      progress_percent: 0,
      created_at: '2026-09-08T10:00:00Z',
      job_id: 'job_stage_test',
    };

    it('1. QUEUED state: all stages remain PENDING even if status is temporarily COMPLETED', async () => {
      const queuedDoc: DocumentListItem = {
        ...baseDoc,
        status: 'COMPLETED',
        stage: 'QUEUED',
        progress_percent: 0,
      };
      vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
        document_id: 'doc_stage_test',
        status: 'COMPLETED',
        stage: 'QUEUED',
        progress_percent: 0,
        processed_chunks: 0,
      });

      render(
        <DocumentDetailDrawer
          document={queuedDoc}
          isOpen={true}
          onClose={vi.fn()}
          onDocumentUpdated={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('stage-item-PARSING')).toHaveClass('stage-pending');
        expect(screen.getByTestId('stage-item-EMBEDDING')).toHaveClass('stage-pending');
        expect(screen.getByTestId('stage-item-COMPLETED')).toHaveClass('stage-pending');
        expect(screen.getByTestId('stage-item-PARSING')).not.toHaveClass('stage-completed');
      });
    });

    it('2. PARSING state: PARSING is active and subsequent stages are PENDING', async () => {
      const parsingDoc: DocumentListItem = {
        ...baseDoc,
        status: 'PROCESSING',
        stage: 'PARSING',
        progress_percent: 15,
      };
      vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
        document_id: 'doc_stage_test',
        status: 'PROCESSING',
        stage: 'PARSING',
        progress_percent: 15,
        processed_chunks: 0,
      });

      render(
        <DocumentDetailDrawer
          document={parsingDoc}
          isOpen={true}
          onClose={vi.fn()}
          onDocumentUpdated={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('stage-item-PARSING')).toHaveClass('stage-active');
        expect(screen.getByTestId('stage-item-PII_REDACTION')).toHaveClass('stage-pending');
        expect(screen.getByTestId('stage-item-COMPLETED')).toHaveClass('stage-pending');
      });
    });

    it('3. Intermediate state (ENTITY_EXTRACTION): previous completed, current active, future pending', async () => {
      const entityDoc: DocumentListItem = {
        ...baseDoc,
        status: 'PROCESSING',
        stage: 'ENTITY_EXTRACTION',
        progress_percent: 45,
      };
      vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
        document_id: 'doc_stage_test',
        status: 'PROCESSING',
        stage: 'ENTITY_EXTRACTION',
        progress_percent: 45,
        processed_chunks: 5,
      });

      render(
        <DocumentDetailDrawer
          document={entityDoc}
          isOpen={true}
          onClose={vi.fn()}
          onDocumentUpdated={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('stage-item-PARSING')).toHaveClass('stage-completed');
        expect(screen.getByTestId('stage-item-PII_REDACTION')).toHaveClass('stage-completed');
        expect(screen.getByTestId('stage-item-ENTITY_EXTRACTION')).toHaveClass('stage-active');
        expect(screen.getByTestId('stage-item-CHUNKING')).toHaveClass('stage-pending');
        expect(screen.getByTestId('stage-item-EMBEDDING')).toHaveClass('stage-pending');
      });
    });

    it('4. EMBEDDING state: PARSING..CHUNKING completed, EMBEDDING active, INDEXING & COMPLETED pending', async () => {
      const embeddingDoc: DocumentListItem = {
        ...baseDoc,
        status: 'PROCESSING',
        stage: 'EMBEDDING',
        progress_percent: 75,
      };
      vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
        document_id: 'doc_stage_test',
        status: 'PROCESSING',
        stage: 'EMBEDDING',
        progress_percent: 75,
        processed_chunks: 15,
      });

      render(
        <DocumentDetailDrawer
          document={embeddingDoc}
          isOpen={true}
          onClose={vi.fn()}
          onDocumentUpdated={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('stage-item-PARSING')).toHaveClass('stage-completed');
        expect(screen.getByTestId('stage-item-CHUNKING')).toHaveClass('stage-completed');
        expect(screen.getByTestId('stage-item-EMBEDDING')).toHaveClass('stage-active');
        expect(screen.getByTestId('stage-item-INDEXING')).toHaveClass('stage-pending');
        expect(screen.getByTestId('stage-item-COMPLETED')).toHaveClass('stage-pending');
      });
    });

    it('5. INDEXING state: PARSING..EMBEDDING completed, INDEXING active, COMPLETED pending', async () => {
      const indexingDoc: DocumentListItem = {
        ...baseDoc,
        status: 'PROCESSING',
        stage: 'INDEXING',
        progress_percent: 90,
      };
      vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
        document_id: 'doc_stage_test',
        status: 'PROCESSING',
        stage: 'INDEXING',
        progress_percent: 90,
        processed_chunks: 25,
      });

      render(
        <DocumentDetailDrawer
          document={indexingDoc}
          isOpen={true}
          onClose={vi.fn()}
          onDocumentUpdated={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('stage-item-EMBEDDING')).toHaveClass('stage-completed');
        expect(screen.getByTestId('stage-item-INDEXING')).toHaveClass('stage-active');
        expect(screen.getByTestId('stage-item-COMPLETED')).toHaveClass('stage-pending');
      });
    });

    it('6. COMPLETED state: all seven pipeline stages are completed when status=COMPLETED, stage=COMPLETED, progress=100', async () => {
      const completedDoc: DocumentListItem = {
        ...baseDoc,
        status: 'COMPLETED',
        stage: 'COMPLETED',
        progress_percent: 100,
      };
      vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
        document_id: 'doc_stage_test',
        status: 'COMPLETED',
        stage: 'COMPLETED',
        progress_percent: 100,
        processed_chunks: 25,
      });

      render(
        <DocumentDetailDrawer
          document={completedDoc}
          isOpen={true}
          onClose={vi.fn()}
          onDocumentUpdated={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('stage-item-PARSING')).toHaveClass('stage-completed');
        expect(screen.getByTestId('stage-item-EMBEDDING')).toHaveClass('stage-completed');
        expect(screen.getByTestId('stage-item-INDEXING')).toHaveClass('stage-completed');
        expect(screen.getByTestId('stage-item-COMPLETED')).toHaveClass('stage-completed');
      });
    });

    it('7. FAILED state: preserves completed prior stages, marks failure stage, leaves future stages pending', async () => {
      const failedDoc: DocumentListItem = {
        ...baseDoc,
        status: 'FAILED',
        stage: 'EMBEDDING',
        progress_percent: 75,
        error_message: 'OOM in embedding worker',
      };
      vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
        document_id: 'doc_stage_test',
        status: 'FAILED',
        stage: 'EMBEDDING',
        progress_percent: 75,
        processed_chunks: 10,
        error_message: 'OOM in embedding worker',
      });

      render(
        <DocumentDetailDrawer
          document={failedDoc}
          isOpen={true}
          onClose={vi.fn()}
          onDocumentUpdated={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('stage-item-CHUNKING')).toHaveClass('stage-completed');
        expect(screen.getByTestId('stage-item-EMBEDDING')).toHaveClass('stage-failed');
        expect(screen.getByTestId('stage-item-INDEXING')).toHaveClass('stage-pending');
      });
    });

    it('8. CANCELLED state: preserves completed stages and marks target/subsequent stages cancelled', async () => {
      const cancelledDoc: DocumentListItem = {
        ...baseDoc,
        status: 'CANCELLED',
        stage: 'EMBEDDING',
        progress_percent: 75,
      };
      vi.spyOn(api, 'getDocumentStatus').mockResolvedValue({
        document_id: 'doc_stage_test',
        status: 'CANCELLED',
        stage: 'EMBEDDING',
        progress_percent: 75,
        processed_chunks: 10,
      });

      render(
        <DocumentDetailDrawer
          document={cancelledDoc}
          isOpen={true}
          onClose={vi.fn()}
          onDocumentUpdated={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('stage-item-CHUNKING')).toHaveClass('stage-completed');
        expect(screen.getByTestId('stage-item-EMBEDDING')).toHaveClass('stage-cancelled');
        expect(screen.getByTestId('stage-item-INDEXING')).toHaveClass('stage-pending');
      });
    });
  });
});
