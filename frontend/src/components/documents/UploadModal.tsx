import React, { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { X, UploadCloud, FileText, AlertCircle, Loader2 } from 'lucide-react';
import { api } from '../../services/api';
import { DocumentStatus } from '../../types/api';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadAccepted: (job: DocumentStatus) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onUploadAccepted,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [customTitle, setCustomTitle] = useState<string>('');
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    setErrorMessage(null);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      validateAndSelectFile(files[0]);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    setErrorMessage(null);
    if (e.target.files && e.target.files.length > 0) {
      validateAndSelectFile(e.target.files[0]);
    }
  };

  const validateAndSelectFile = (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setErrorMessage('Invalid file format. Only .pdf files are supported.');
      setSelectedFile(null);
      return;
    }
    if (file.size === 0) {
      setErrorMessage('The selected file is empty (0 bytes).');
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
    if (!customTitle) {
      setCustomTitle(file.name.replace(/\.pdf$/i, ''));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      // POST /documents returns 202 Accepted with job_id and document_id
      const statusRes = await api.uploadDocument(selectedFile, customTitle);
      onUploadAccepted(statusRes);
      handleClose();
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to upload document.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    setSelectedFile(null);
    setCustomTitle('');
    setErrorMessage(null);
    setIsSubmitting(false);
    onClose();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="modal-backdrop" onClick={handleClose} data-testid="upload-modal-backdrop">
      <div 
        className="modal-card" 
        onClick={(e) => e.stopPropagation()} 
        role="dialog" 
        aria-labelledby="upload-title"
        data-testid="upload-modal"
      >
        <div className="modal-header">
          <h3 id="upload-title">Upload Document</h3>
          <button 
            className="btn btn-secondary btn-sm" 
            onClick={handleClose} 
            disabled={isSubmitting}
            aria-label="Close dialog"
          >
            <X size={15} />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            {errorMessage && (
              <div 
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  backgroundColor: 'var(--status-failed-bg)',
                  border: '1px solid var(--status-failed-border)',
                  color: 'var(--status-failed-text)',
                  fontSize: '12.5px',
                }}
                data-testid="upload-error-message"
              >
                <AlertCircle size={15} flex-shrink="0" />
                <span>{errorMessage}</span>
              </div>
            )}

            {!selectedFile ? (
              <div
                className={`dropzone ${isDragging ? 'active' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                data-testid="file-dropzone"
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept=".pdf,application/pdf"
                  style={{ display: 'none' }}
                  data-testid="file-input"
                />
                <div className="dropzone-icon">
                  <UploadCloud size={32} />
                </div>
                <div className="dropzone-title">Click to upload or drag & drop</div>
                <div className="dropzone-subtitle">PDF files only (Max 50MB)</div>
              </div>
            ) : (
              <div className="selected-file-card" data-testid="selected-file-card">
                <FileText size={22} color="#2563eb" />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '13px', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {selectedFile.name}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    {formatFileSize(selectedFile.size)}
                  </div>
                </div>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => setSelectedFile(null)}
                  disabled={isSubmitting}
                  aria-label="Remove selected file"
                >
                  <X size={13} />
                </button>
              </div>
            )}

            {selectedFile && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <label style={{ fontSize: '11.5px', fontWeight: 500, color: 'var(--text-secondary)' }}>
                  Document Title (Optional)
                </label>
                <input
                  type="text"
                  className="search-input"
                  style={{ height: '36px', padding: '0 12px' }}
                  placeholder={selectedFile.name}
                  value={customTitle}
                  onChange={(e) => setCustomTitle(e.target.value)}
                  disabled={isSubmitting}
                  data-testid="title-input"
                />
              </div>
            )}
          </div>

          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleClose}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={!selectedFile || isSubmitting}
              data-testid="submit-upload-btn"
            >
              {isSubmitting ? (
                <>
                  <Loader2 size={14} className="loading-pulse" />
                  <span>Uploading...</span>
                </>
              ) : (
                <span>Upload & Ingest</span>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
