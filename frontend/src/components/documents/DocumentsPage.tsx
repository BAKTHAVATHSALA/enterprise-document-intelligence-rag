import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Plus, 
  FileText, 
  RefreshCw, 
  Layers, 
  AlertCircle,
  Inbox,
  Trash2
} from 'lucide-react';
import { api } from '../../services/api';
import { DocumentListItem, DocumentStatus } from '../../types/api';
import { UploadModal } from './UploadModal';
import { DocumentDetailDrawer } from '../ingestion/DocumentDetailDrawer';

interface DocumentsPageProps {
  onDocumentCountChange?: (count: number) => void;
}

export const DocumentsPage: React.FC<DocumentsPageProps> = ({ onDocumentCountChange }) => {
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [selectedDoc, setSelectedDoc] = useState<DocumentListItem | null>(null);
  const [isDetailDrawerOpen, setIsDetailDrawerOpen] = useState<boolean>(false);

  // Delete modal state
  const [deleteConfirmDoc, setDeleteConfirmDoc] = useState<DocumentListItem | null>(null);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Polling tracker ref to prevent concurrent requests
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isMountedRef = useRef<boolean>(true);

  const fetchDocuments = useCallback(async () => {
    try {
      setError(null);
      const docs = await api.listDocuments();
      if (isMountedRef.current) {
        setDocuments(docs);
        if (onDocumentCountChange) {
          onDocumentCountChange(docs.length);
        }
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        setError(err.message || 'Failed to load documents from database.');
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [onDocumentCountChange]);

  useEffect(() => {
    isMountedRef.current = true;
    fetchDocuments();

    return () => {
      isMountedRef.current = false;
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [fetchDocuments]);

  // Polling loop for active (non-terminal) documents
  useEffect(() => {
    const activeDocs = documents.filter(
      d => d.status === 'PENDING' || d.status === 'PROCESSING'
    );

    if (activeDocs.length === 0) {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }

    if (!pollingRef.current) {
      pollingRef.current = setInterval(async () => {
        if (!isMountedRef.current) return;

        // Poll status for each active document
        for (const doc of activeDocs) {
          try {
            const statusUpdate = await api.getDocumentStatus(doc.document_id);
            if (!isMountedRef.current) return;

            setDocuments(prevDocs => 
              prevDocs.map(item => {
                if (item.document_id === doc.document_id) {
                  return {
                    ...item,
                    status: statusUpdate.status,
                    stage: statusUpdate.stage || item.stage,
                    progress_percent: statusUpdate.progress_percent ?? item.progress_percent,
                    total_chunks: statusUpdate.processed_chunks || item.total_chunks,
                    error_message: statusUpdate.error_message || item.error_message,
                  };
                }
                return item;
              })
            );

            // If selected document in drawer is updating, sync it
            if (selectedDoc && selectedDoc.document_id === doc.document_id) {
              setSelectedDoc(prev => prev ? {
                ...prev,
                status: statusUpdate.status,
                stage: statusUpdate.stage || prev.stage,
                progress_percent: statusUpdate.progress_percent ?? prev.progress_percent,
                total_chunks: statusUpdate.processed_chunks || prev.total_chunks,
                error_message: statusUpdate.error_message || prev.error_message,
              } : null);
            }
          } catch {
            // Ignore polling network hiccup
          }
        }
      }, 1500);
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [documents, selectedDoc]);

  const handleUploadAccepted = (statusRes: DocumentStatus) => {
    const newDoc: DocumentListItem = {
      document_id: statusRes.document_id,
      tenant_id: statusRes.tenant_id || 'default_tenant',
      title: statusRes.document_id,
      source_filename: 'uploaded_document.pdf',
      total_pages: 1,
      total_chunks: 0,
      status: statusRes.status,
      stage: statusRes.stage || 'QUEUED',
      progress_percent: statusRes.progress_percent || 0,
      created_at: new Date().toISOString(),
      job_id: statusRes.job_id,
    };

    setDocuments(prev => [newDoc, ...prev]);
    if (onDocumentCountChange) {
      onDocumentCountChange(documents.length + 1);
    }
    // Refresh to get full metadata from PostgreSQL
    setTimeout(() => {
      fetchDocuments();
    }, 500);
  };

  const handleRowClick = (doc: DocumentListItem) => {
    setSelectedDoc(doc);
    setIsDetailDrawerOpen(true);
  };

  const handleDocumentUpdated = (updated: DocumentListItem) => {
    setDocuments(prev => prev.map(d => d.document_id === updated.document_id ? updated : d));
    setSelectedDoc(updated);
  };

  const handleDeleteClick = (e: React.MouseEvent, doc: DocumentListItem) => {
    e.stopPropagation();
    setDeleteConfirmDoc(doc);
    setDeleteError(null);
  };

  const handleConfirmDelete = async () => {
    if (!deleteConfirmDoc) return;
    setIsDeleting(true);
    setDeleteError(null);

    try {
      await api.deleteDocument(deleteConfirmDoc.document_id);
      if (selectedDoc && selectedDoc.document_id === deleteConfirmDoc.document_id) {
        setIsDetailDrawerOpen(false);
        setSelectedDoc(null);
      }
      setDeleteConfirmDoc(null);
      await fetchDocuments();
    } catch (err: any) {
      setDeleteError(err.message || 'Failed to delete document.');
    } finally {
      setIsDeleting(false);
    }
  };

  const formatDate = (dateStr: string) => {
    try {
      const d = new Date(dateStr);
      return d.toLocaleDateString(undefined, { 
        month: 'short', 
        day: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit' 
      });
    } catch {
      return dateStr;
    }
  };

  return (
    <>
      <div className="page-header" data-testid="documents-page-header">
        <div className="page-header-text">
          <h1>Documents</h1>
          <p>Manage uploaded documents and monitor ingestion status.</p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            className="btn btn-secondary btn-sm"
            onClick={fetchDocuments}
            aria-label="Refresh document list"
            data-testid="refresh-docs-btn"
          >
            <RefreshCw size={13} />
          </button>
          <button
            className="btn btn-primary"
            onClick={() => setIsUploadModalOpen(true)}
            data-testid="upload-document-btn"
          >
            <Plus size={15} />
            <span>Upload document</span>
          </button>
        </div>
      </div>

      <div className="page-content" data-testid="documents-page-content">
        {error && (
          <div 
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '12px 16px',
              borderRadius: '8px',
              backgroundColor: 'var(--status-failed-bg)',
              border: '1px solid var(--status-failed-border)',
              color: 'var(--status-failed-text)',
              fontSize: '13px',
              marginBottom: '16px',
            }}
            data-testid="documents-error-alert"
          >
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {isLoading ? (
          <div className="state-container" data-testid="documents-loading">
            <Layers size={32} className="state-icon loading-pulse" />
            <div className="state-title">Loading documents...</div>
            <div className="state-description">Fetching indexed enterprise document repository.</div>
          </div>
        ) : documents.length === 0 ? (
          <div className="state-container" data-testid="documents-empty-state">
            <Inbox size={40} className="state-icon" />
            <div className="state-title">No documents indexed yet</div>
            <div className="state-description">
              Upload your first enterprise PDF to extract text, redact PII, build knowledge graphs, and enable Hybrid RAG search.
            </div>
            <button
              className="btn btn-primary"
              style={{ marginTop: '8px' }}
              onClick={() => setIsUploadModalOpen(true)}
              data-testid="empty-upload-btn"
            >
              <Plus size={15} />
              <span>Upload document</span>
            </button>
          </div>
        ) : (
          <div className="table-container" data-testid="documents-table-container">
            <table className="data-table" data-testid="documents-table">
              <thead>
                <tr>
                  <th style={{ width: '30%' }}>Document</th>
                  <th style={{ width: '13%' }}>Status</th>
                  <th style={{ width: '14%' }}>Stage</th>
                  <th style={{ width: '15%' }}>Progress</th>
                  <th style={{ width: '7%', textAlign: 'center' }}>Pages</th>
                  <th style={{ width: '7%', textAlign: 'center' }}>Chunks</th>
                  <th style={{ width: '8%', textAlign: 'right' }}>Updated</th>
                  <th style={{ width: '6%', textAlign: 'center' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => {
                  const statusVal = doc.status || 'PENDING';
                  const stageVal = doc.stage || (statusVal === 'COMPLETED' ? 'COMPLETED' : 'QUEUED');
                  const progressVal = doc.progress_percent !== undefined ? doc.progress_percent : (statusVal === 'COMPLETED' ? 100 : 0);

                  return (
                    <tr 
                      key={doc.document_id} 
                      onClick={() => handleRowClick(doc)}
                      data-testid={`document-row-${doc.document_id}`}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>
                        <div className="doc-name-cell">
                          <FileText size={18} className="doc-icon" color="#3b82f6" />
                          <div className="doc-info">
                            <span className="doc-title">{doc.title || doc.source_filename}</span>
                            <span className="doc-id">{doc.document_id}</span>
                          </div>
                        </div>
                      </td>
                      <td>
                        <span className={`status-badge status-${statusVal.toLowerCase()}`}>
                          <span className="dot" />
                          <span>{statusVal}</span>
                        </span>
                      </td>
                      <td>
                        <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                          {stageVal}
                        </span>
                      </td>
                      <td>
                        <div className="progress-cell">
                          <div className="progress-bar-bg">
                            <div 
                              className={`progress-bar-fill ${statusVal === 'COMPLETED' ? 'completed' : ''}`}
                              style={{ width: `${Math.min(100, Math.max(0, progressVal))}%` }}
                            />
                          </div>
                          <span className="progress-text">{Math.round(progressVal)}%</span>
                        </div>
                      </td>
                      <td style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                        {doc.total_pages || 1}
                      </td>
                      <td style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                        {doc.total_chunks || 0}
                      </td>
                      <td style={{ textAlign: 'right', fontSize: '11.5px', color: 'var(--text-muted)' }}>
                        {formatDate(doc.created_at)}
                      </td>
                      <td style={{ textAlign: 'center' }} onClick={(e) => e.stopPropagation()}>
                        <button
                          className="btn btn-secondary btn-sm"
                          style={{ color: '#ef4444', padding: '4px 8px' }}
                          onClick={(e) => handleDeleteClick(e, doc)}
                          title="Delete document"
                          aria-label={`Delete document ${doc.document_id}`}
                          data-testid={`delete-doc-btn-${doc.document_id}`}
                        >
                          <Trash2 size={13} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {deleteConfirmDoc && (
        <div 
          className="modal-overlay" 
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            backgroundColor: 'rgba(0,0,0,0.6)', 
            position: 'fixed', 
            inset: 0, 
            zIndex: 1000 
          }}
          onClick={() => { setDeleteConfirmDoc(null); setDeleteError(null); }}
        >
          <div 
            className="modal-card" 
            style={{ 
              maxWidth: '460px', 
              width: '90%', 
              padding: '24px', 
              borderRadius: '12px', 
              background: 'var(--bg-secondary, #1e293b)', 
              border: '1px solid var(--border-color, #334155)', 
              color: '#fff',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)' 
            }}
            onClick={(e) => e.stopPropagation()}
            data-testid="delete-confirm-modal"
          >
            <h3 style={{ margin: '0 0 12px 0', fontSize: '18px', fontWeight: 600, color: '#f8fafc' }}>
              Delete Document?
            </h3>
            <p style={{ fontSize: '13.5px', color: '#94a3b8', marginBottom: '20px', lineHeight: '1.5' }}>
              Are you sure you want to delete <strong>{deleteConfirmDoc.title || deleteConfirmDoc.source_filename}</strong> (<code>{deleteConfirmDoc.document_id}</code>)?
              This action will permanently purge the document and all associated chunks, vectors, graph nodes, and BM25 index entries.
            </p>
            {deleteError && (
              <div style={{ color: '#ef4444', fontSize: '13px', marginBottom: '16px', backgroundColor: 'rgba(239,68,68,0.1)', padding: '8px 12px', borderRadius: '6px' }}>
                {deleteError}
              </div>
            )}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button
                className="btn btn-secondary"
                onClick={() => { setDeleteConfirmDoc(null); setDeleteError(null); }}
                disabled={isDeleting}
              >
                Cancel
              </button>
              <button
                className="btn"
                style={{ backgroundColor: '#dc2626', color: '#fff', border: 'none', padding: '8px 16px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 500 }}
                onClick={handleConfirmDelete}
                disabled={isDeleting}
                data-testid="confirm-delete-btn"
              >
                <Trash2 size={14} />
                <span>{isDeleting ? 'Deleting...' : 'Delete Document'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUploadAccepted={handleUploadAccepted}
      />

      <DocumentDetailDrawer
        document={selectedDoc}
        isOpen={isDetailDrawerOpen}
        onClose={() => setIsDetailDrawerOpen(false)}
        onDocumentUpdated={handleDocumentUpdated}
      />
    </>
  );
};

