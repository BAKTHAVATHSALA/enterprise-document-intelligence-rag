import React, { useState, FormEvent } from 'react';
import { 
  Search, 
  FileText, 
  AlertCircle, 
  Clock, 
  ShieldCheck, 
  HelpCircle, 
  Loader2,
  Filter,
  Sparkles
} from 'lucide-react';
import { api } from '../../services/api';
import { 
  DocumentListItem, 
  QueryResponse, 
  Citation 
} from '../../types/api';

interface KnowledgeSearchPageProps {
  documents: DocumentListItem[];
}

interface SearchHistoryItem {
  id: string;
  question: string;
  response: QueryResponse;
  timestamp: Date;
}

export const KnowledgeSearchPage: React.FC<KnowledgeSearchPageProps> = ({ documents }) => {
  const [queryText, setQueryText] = useState<string>('');
  const [selectedDocId, setSelectedDocId] = useState<string>('all');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [history, setHistory] = useState<SearchHistoryItem[]>([]);

  const handleSearch = async (e: FormEvent) => {
    e.preventDefault();
    const cleanQuery = queryText.trim();
    if (!cleanQuery || isLoading) return;

    setIsLoading(true);
    setErrorMessage(null);

    const docFilter = selectedDocId === 'all' ? null : [selectedDocId];

    try {
      const response = await api.searchKnowledge({
        question: cleanQuery,
        document_ids: docFilter,
        top_k: 5,
      });

      const historyItem: SearchHistoryItem = {
        id: `search_${Date.now()}`,
        question: cleanQuery,
        response,
        timestamp: new Date(),
      };

      setHistory(prev => [historyItem, ...prev]);
      setQueryText('');
    } catch (err: any) {
      setErrorMessage(err.message || 'Something went wrong while searching. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const completedDocs = documents.filter(d => d.status === 'COMPLETED');

  // Format inline citation text
  const renderFormattedAnswer = (text: string) => {
    // Regex matching bracketed citations like [Doc: ..., Page: ..., Section: ...]
    const citationRegex = /(\[Doc:\s*[^,]+,\s*Page:\s*\d+,\s*Section:\s*[^\]]+\]|\[Page\s*\d+,\s*[^\]]+\]|\[[^\]]+,\s*Page\s*\d+\])/g;
    const parts = text.split(citationRegex);

    return parts.map((part, index) => {
      if (citationRegex.test(part)) {
        return (
          <span key={index} className="inline-citation-tag" data-testid="inline-citation">
            {part}
          </span>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <>
      <div className="page-header" data-testid="search-page-header">
        <div className="page-header-text">
          <h1>Knowledge Search</h1>
          <p>Search across your indexed enterprise documents with verifiable citations.</p>
        </div>
      </div>

      <div className="page-content" data-testid="search-page-content">
        <div className="search-container">
          {/* Main Search Bar Card */}
          <div className="search-bar-card">
            <form onSubmit={handleSearch}>
              <div className="search-input-row">
                <div className="search-input-wrapper">
                  <Search size={18} className="search-icon-inside" />
                  <input
                    type="text"
                    className="search-input"
                    placeholder="Ask a question about your indexed documents..."
                    value={queryText}
                    onChange={(e) => setQueryText(e.target.value)}
                    disabled={isLoading || documents.length === 0}
                    data-testid="search-input"
                  />
                </div>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={!queryText.trim() || isLoading || documents.length === 0}
                  data-testid="search-submit-btn"
                >
                  {isLoading ? (
                    <>
                      <Loader2 size={14} className="loading-pulse" />
                      <span>Searching...</span>
                    </>
                  ) : (
                    <span>Search</span>
                  )}
                </button>
              </div>

              <div className="search-controls-row">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Filter size={13} color="#64748b" />
                  <label htmlFor="doc-filter" style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                    Filter Document:
                  </label>
                  <select
                    id="doc-filter"
                    className="filter-select"
                    value={selectedDocId}
                    onChange={(e) => setSelectedDocId(e.target.value)}
                    disabled={isLoading || documents.length === 0}
                    data-testid="document-filter-select"
                  >
                    <option value="all">All Indexed Documents ({completedDocs.length})</option>
                    {documents.map((doc) => (
                      <option key={doc.document_id} value={doc.document_id}>
                        {doc.title || doc.source_filename}
                      </option>
                    ))}
                  </select>
                </div>
                <div style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                  Hybrid Vector + BM25 + Graph RAG
                </div>
              </div>
            </form>
          </div>

          {/* Error State */}
          {errorMessage && (
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
              }}
              data-testid="search-error-alert"
            >
              <AlertCircle size={16} />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Empty State (No Documents Indexed) */}
          {documents.length === 0 && (
            <div className="state-container" data-testid="search-no-docs-state">
              <FileText size={36} className="state-icon" />
              <div className="state-title">No documents have been indexed yet</div>
              <div className="state-description">
                Upload and index a PDF document from the Documents page to begin asking grounded questions.
              </div>
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="state-container" data-testid="search-loading-state">
              <Search size={32} className="state-icon loading-pulse" />
              <div className="state-title">Searching your enterprise knowledge...</div>
              <div className="state-description">
                Retrieving dense vector embeddings, BM25 inverted indexes, and Neo4j entity graphs.
              </div>
            </div>
          )}

          {/* Search History / Results List */}
          {!isLoading && history.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {history.map((item) => {
                const resp = item.response;
                const isInsufficient = resp.answer.toLowerCase().includes('insufficient') || 
                                       resp.answer.toLowerCase().includes('does not provide') ||
                                       resp.answer.toLowerCase().includes('not mentioned');

                return (
                  <div key={item.id} className="answer-card" data-testid="answer-card">
                    <div className="answer-header">
                      <div className="answer-question" data-testid="answer-question">
                        {item.question}
                      </div>
                      <div className="answer-badge-group">
                        {resp.confidence_score !== undefined && (
                          <span 
                            className={`metric-pill ${resp.confidence_score >= 0.8 ? 'confidence-high' : ''}`}
                            data-testid="confidence-pill"
                          >
                            <ShieldCheck size={12} />
                            <span>Confidence: {Math.round(resp.confidence_score * 100)}%</span>
                          </span>
                        )}
                        {resp.processing_time_ms !== undefined && (
                          <span className="metric-pill" data-testid="latency-pill">
                            <Clock size={12} />
                            <span>{resp.processing_time_ms}ms</span>
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="answer-body" data-testid="answer-body">
                      {isInsufficient ? (
                        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', color: 'var(--text-secondary)' }}>
                          <HelpCircle size={18} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
                          <span>{resp.answer}</span>
                        </div>
                      ) : (
                        renderFormattedAnswer(resp.answer)
                      )}
                    </div>

                    {/* Sources & Citations Breakdown */}
                    {resp.citations && resp.citations.length > 0 && (
                      <div className="sources-section" data-testid="sources-section">
                        <div className="sources-title">
                          Sources & Evidence ({resp.citations.length})
                        </div>
                        <div className="sources-grid">
                          {resp.citations.map((cite: Citation, cIdx: number) => (
                            <div key={cIdx} className="source-card" data-testid={`source-card-${cIdx}`}>
                              <div className="source-card-header">
                                <span className="source-filename">
                                  <FileText size={14} color="#2563eb" />
                                  <span>{cite.source || cite.document_id}</span>
                                </span>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                  <span className="source-location-tag">
                                    Page {cite.page} · {cite.section}
                                  </span>
                                  <span 
                                    className={`status-badge status-${cite.validation_status ? cite.validation_status.toLowerCase() : 'completed'}`}
                                    style={{ fontSize: '10.5px', padding: '1px 6px' }}
                                    data-testid={`citation-status-${cIdx}`}
                                  >
                                    {cite.validation_status || 'VALID'}
                                  </span>
                                </div>
                              </div>
                              {cite.snippet && (
                                <div className="source-snippet">
                                  "{cite.snippet}"
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Initial Clean Workspace State (When no queries run yet) */}
          {!isLoading && history.length === 0 && documents.length > 0 && (
            <div className="state-container" style={{ border: 'none', background: 'transparent' }} data-testid="search-initial-state">
              <Sparkles size={32} color="#94a3b8" />
              <div className="state-title" style={{ color: 'var(--text-secondary)' }}>
                Ask anything across your {completedDocs.length} indexed documents
              </div>
              <div className="state-description">
                Answers are grounded strictly in retrieved facts and include verifiable page and section citations.
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};
