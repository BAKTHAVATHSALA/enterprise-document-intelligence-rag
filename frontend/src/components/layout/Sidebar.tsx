import React from 'react';
import { FileText, Search, Activity, Layers } from 'lucide-react';

export type NavView = 'documents' | 'search';

interface SidebarProps {
  currentView: NavView;
  onSelectView: (view: NavView) => void;
  documentCount: number;
  apiHealthy: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  onSelectView,
  documentCount,
  apiHealthy,
}) => {
  return (
    <aside className="app-sidebar" data-testid="app-sidebar">
      <div className="app-sidebar-header">
        <div className="brand-badge">
          <Layers size={16} />
        </div>
        <span className="brand-title">Document Intelligence</span>
        <span className="brand-version">v1.0</span>
      </div>

      <div className="app-sidebar-content">
        <div className="nav-group">
          <div className="nav-group-label">Documents</div>
          <button
            className={`nav-item ${currentView === 'documents' ? 'active' : ''}`}
            onClick={() => onSelectView('documents')}
            data-testid="nav-documents"
          >
            <FileText size={16} />
            <span>Documents</span>
            {documentCount > 0 && (
              <span className="nav-item-count" data-testid="sidebar-doc-count">{documentCount}</span>
            )}
          </button>
        </div>

        <div className="nav-group">
          <div className="nav-group-label">Search</div>
          <button
            className={`nav-item ${currentView === 'search' ? 'active' : ''}`}
            onClick={() => onSelectView('search')}
            data-testid="nav-search"
          >
            <Search size={16} />
            <span>Knowledge Search</span>
          </button>
        </div>
      </div>

      <div className="app-sidebar-footer">
        <div className="health-indicator" data-testid="health-status">
          <div className={`health-dot ${apiHealthy ? '' : 'offline'}`} />
          <span>{apiHealthy ? 'API Connected' : 'API Disconnected'}</span>
        </div>
        <Activity size={13} color="#64748b" />
      </div>
    </aside>
  );
};
