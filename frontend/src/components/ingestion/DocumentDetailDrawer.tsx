import React, { useState, useEffect } from 'react';
import { 
  X, 
  FileText, 
  CheckCircle2, 
  Clock, 
  AlertTriangle, 
  XCircle, 
  Ban,
  ArrowRight,
  Loader2,
} from 'lucide-react';
import { api } from '../../services/api';
import { 
  DocumentListItem, 
  DocumentStatus, 
  IngestionStageEnum 
} from '../../types/api';

interface DocumentDetailDrawerProps {
  document: DocumentListItem | null;
  isOpen: boolean;
  onClose: () => void;
  onDocumentUpdated: (updated: DocumentListItem) => void;
}

interface StageStep {
  stage: IngestionStageEnum;
  label: string;
  nominalPercent: number;
}

const PIPELINE_STAGES: StageStep[] = [
  { stage: 'PARSING', label: 'Layout & PDF Parsing', nominalPercent: 15 },
  { stage: 'PII_REDACTION', label: 'PII Redaction', nominalPercent: 30 },
  { stage: 'ENTITY_EXTRACTION', label: 'Entity Extraction', nominalPercent: 45 },
  { stage: 'CHUNKING', label: 'Semantic Chunking', nominalPercent: 60 },
  { stage: 'EMBEDDING', label: 'Vector Embedding', nominalPercent: 75 },
  { stage: 'INDEXING', label: 'Multi-Store Indexing', nominalPercent: 90 },
  { stage: 'COMPLETED', label: 'Completed', nominalPercent: 100 },
];

export const DocumentDetailDrawer: React.FC<DocumentDetailDrawerProps> = ({
  document,
  isOpen,
  onClose,
  onDocumentUpdated,
}) => {
  const [currentStatus, setCurrentStatus] = useState<DocumentStatus | null>(null);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  useEffect(() => {
    if (document) {
      // Sync initial state from document
      setCurrentStatus({
        document_id: document.document_id,
        job_id: document.job_id,
        status: document.status,
        stage: document.stage || null,
        progress_percent: document.progress_percent || 0,
        processed_chunks: document.total_chunks || 0,
        error_message: document.error_message,
      });

      // Fetch fresh status from backend
      api.getDocumentStatus(document.document_id)
        .then((fresh) => {
          setCurrentStatus(fresh);
        })
        .catch(() => {
          // Keep initial state if fetch fails
        });
    }
  }, [document]);

  if (!isOpen || !document) return null;

  const statusVal = currentStatus?.status || document.status;
  const stageVal = currentStatus?.stage ?? document.stage ?? (statusVal === 'COMPLETED' ? 'COMPLETED' : 'QUEUED');
  const progressPercent = currentStatus?.progress_percent ?? document.progress_percent ?? (statusVal === 'COMPLETED' && stageVal === 'COMPLETED' ? 100 : 0);
  const isProcessing = ['PENDING', 'PROCESSING'].includes(statusVal);

  const handleCancel = async () => {
    if (!document) return;
    setIsCancelling(true);
    setCancelError(null);

    try {
      await api.cancelDocument(document.document_id);
      const updated: DocumentListItem = {
        ...document,
        status: 'CANCELLED',
        stage: 'CANCELLED',
        error_message: 'Job cancelled by user request.',
      };
      setCurrentStatus((prev) => prev ? { ...prev, status: 'CANCELLED', stage: 'CANCELLED' } : null);
      onDocumentUpdated(updated);
    } catch (err: any) {
      setCancelError(err.message || 'Failed to cancel ingestion job.');
    } finally {
      setIsCancelling(false);
    }
  };

  const getStageStatus = (stageStep: StageStep, currentStageName: string) => {
    // Only when status=COMPLETED, stage=COMPLETED, and progress>=100 are all stages marked completed
    if (statusVal === 'COMPLETED' && currentStageName === 'COMPLETED' && progressPercent >= 100) {
      return 'completed';
    }

    if (statusVal === 'CANCELLED' || currentStageName === 'CANCELLED') {
      const currentIndex = PIPELINE_STAGES.findIndex(s => s.stage === currentStageName);
      const thisIndex = PIPELINE_STAGES.findIndex(s => s.stage === stageStep.stage);
      if (currentIndex >= 0) {
        if (thisIndex < currentIndex) return 'completed';
        if (thisIndex === currentIndex) return 'cancelled';
        return 'pending';
      }
      if (stageStep.nominalPercent <= progressPercent) return 'completed';
      return 'cancelled';
    }

    if (statusVal === 'FAILED' || currentStageName === 'FAILED') {
      const currentIndex = PIPELINE_STAGES.findIndex(s => s.stage === currentStageName);
      const thisIndex = PIPELINE_STAGES.findIndex(s => s.stage === stageStep.stage);
      if (currentIndex >= 0) {
        if (thisIndex < currentIndex) return 'completed';
        if (thisIndex === currentIndex) return 'failed';
        return 'pending';
      }
      if (stageStep.nominalPercent < progressPercent) return 'completed';
      if (stageStep.nominalPercent <= progressPercent + 15) return 'failed';
      return 'pending';
    }

    if (currentStageName === 'QUEUED') {
      return 'pending';
    }

    if (stageStep.stage === currentStageName) {
      return 'active';
    }

    const currentIndex = PIPELINE_STAGES.findIndex(s => s.stage === currentStageName);
    const thisIndex = PIPELINE_STAGES.findIndex(s => s.stage === stageStep.stage);

    if (currentIndex >= 0 && thisIndex < currentIndex) {
      return 'completed';
    }
    return 'pending';
  };

  return (
    <div className="drawer-backdrop" onClick={onClose} data-testid="drawer-backdrop">
      <div 
        className="drawer-panel" 
        onClick={(e) => e.stopPropagation()} 
        role="dialog" 
        aria-labelledby="drawer-title"
        data-testid="document-detail-drawer"
      >
        <div className="drawer-header">
          <div className="drawer-header-title">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <FileText size={18} color="#2563eb" />
              <h2 id="drawer-title">{document.title || document.source_filename}</h2>
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              {document.source_filename}
            </div>
          </div>
          <button 
            className="btn btn-secondary btn-sm" 
            onClick={onClose} 
            aria-label="Close details"
            data-testid="close-drawer-btn"
          >
            <X size={15} />
          </button>
        </div>

        <div className="drawer-content">
          {/* Metadata Section */}
          <div>
            <div className="drawer-section-title">Document Information</div>
            <div className="meta-grid">
              <div className="meta-item">
                <span className="meta-item-label">Document ID</span>
                <span className="meta-item-value" style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px' }} data-testid="drawer-doc-id">
                  {document.document_id}
                </span>
              </div>
              <div className="meta-item">
                <span className="meta-item-label">Status</span>
                <span className={`status-badge status-${statusVal.toLowerCase()}`} data-testid="drawer-status-badge">
                  <span className="dot" />
                  <span>{statusVal}</span>
                </span>
              </div>
              <div className="meta-item">
                <span className="meta-item-label">Pages</span>
                <span className="meta-item-value">{document.total_pages || 1}</span>
              </div>
              <div className="meta-item">
                <span className="meta-item-label">Indexed Chunks</span>
                <span className="meta-item-value">{currentStatus?.processed_chunks ?? document.total_chunks ?? 0}</span>
              </div>
              <div className="meta-item" style={{ gridColumn: 'span 2' }}>
                <span className="meta-item-label">Ingestion Progress</span>
                <span className="meta-item-value">{Math.round(progressPercent)}%</span>
              </div>
            </div>
          </div>

          {/* Cancellation Error Banner */}
          {cancelError && (
            <div 
              style={{
                padding: '8px 12px',
                borderRadius: '6px',
                backgroundColor: 'var(--status-failed-bg)',
                border: '1px solid var(--status-failed-border)',
                color: 'var(--status-failed-text)',
                fontSize: '12px',
              }}
            >
              {cancelError}
            </div>
          )}

          {/* Error Message if Failed */}
          {currentStatus?.error_message && (
            <div 
              style={{
                padding: '10px 12px',
                borderRadius: '6px',
                backgroundColor: 'var(--status-failed-bg)',
                border: '1px solid var(--status-failed-border)',
                color: 'var(--status-failed-text)',
                fontSize: '12.5px',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '8px',
              }}
              data-testid="drawer-error-banner"
            >
              <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <div style={{ fontWeight: 600 }}>Ingestion Failure</div>
                <div>{currentStatus.error_message}</div>
              </div>
            </div>
          )}

          {/* 7-Stage Pipeline Tracker */}
          <div>
            <div className="drawer-section-title">Processing Pipeline</div>
            <div className="pipeline-tracker" data-testid="pipeline-tracker">
              {PIPELINE_STAGES.map((step) => {
                const stageStatus = getStageStatus(step, stageVal);
                return (
                  <div 
                    key={step.stage} 
                    className={`stage-item stage-${stageStatus}`}
                    data-testid={`stage-item-${step.stage}`}
                  >
                    <div className="stage-left">
                      {stageStatus === 'completed' && <CheckCircle2 size={15} color="#16a34a" />}
                      {stageStatus === 'active' && <ArrowRight size={15} color="#2563eb" />}
                      {stageStatus === 'pending' && <Clock size={15} color="#94a3b8" />}
                      {stageStatus === 'failed' && <XCircle size={15} color="#dc2626" />}
                      {stageStatus === 'cancelled' && <Ban size={15} color="#9ca3af" />}
                      <span>{step.label}</span>
                    </div>
                    <div className="stage-right">
                      {stageStatus === 'active' && `${Math.round(progressPercent)}%`}
                      {stageStatus === 'completed' && '100%'}
                      {stageStatus === 'pending' && `${step.nominalPercent}%`}
                      {stageStatus === 'failed' && 'Failed'}
                      {stageStatus === 'cancelled' && 'Cancelled'}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Action Footer */}
          {isProcessing && (
            <div style={{ marginTop: 'auto', paddingTop: '16px' }}>
              <button
                className="btn btn-danger-outline"
                style={{ width: '100%' }}
                onClick={handleCancel}
                disabled={isCancelling}
                data-testid="cancel-ingestion-btn"
              >
                {isCancelling ? (
                  <>
                    <Loader2 size={14} className="loading-pulse" />
                    <span>Cancelling Ingestion...</span>
                  </>
                ) : (
                  <>
                    <Ban size={14} />
                    <span>Cancel Ingestion</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
