import React, { useState, useEffect } from 'react';
import { AppShell } from './components/layout/AppShell';
import { NavView } from './components/layout/Sidebar';
import { DocumentsPage } from './components/documents/DocumentsPage';
import { KnowledgeSearchPage } from './components/search/KnowledgeSearchPage';
import { api } from './services/api';
import { DocumentListItem } from './types/api';

export const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<NavView>('documents');
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [apiHealthy, setApiHealthy] = useState<boolean>(true);

  const checkHealth = async () => {
    try {
      await api.getHealth();
      setApiHealthy(true);
    } catch {
      setApiHealthy(false);
    }
  };

  const refreshDocuments = async () => {
    try {
      const docs = await api.listDocuments();
      setDocuments(docs);
    } catch {
      // Ignore on error
    }
  };

  useEffect(() => {
    checkHealth();
    refreshDocuments();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <AppShell
      currentView={currentView}
      onSelectView={(view) => {
        setCurrentView(view);
        if (view === 'search') {
          refreshDocuments();
        }
      }}
      documentCount={documents.length}
      apiHealthy={apiHealthy}
    >
      {currentView === 'documents' ? (
        <DocumentsPage onDocumentCountChange={() => {
          refreshDocuments();
        }} />
      ) : (
        <KnowledgeSearchPage documents={documents} />
      )}
    </AppShell>
  );
};

export default App;
