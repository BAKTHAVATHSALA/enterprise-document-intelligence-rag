import React, { ReactNode } from 'react';
import { Sidebar, NavView } from './Sidebar';

interface AppShellProps {
  currentView: NavView;
  onSelectView: (view: NavView) => void;
  documentCount: number;
  apiHealthy: boolean;
  children: ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({
  currentView,
  onSelectView,
  documentCount,
  apiHealthy,
  children,
}) => {
  return (
    <div className="app-shell" data-testid="app-shell">
      <Sidebar
        currentView={currentView}
        onSelectView={onSelectView}
        documentCount={documentCount}
        apiHealthy={apiHealthy}
      />
      <main className="app-main">
        {children}
      </main>
    </div>
  );
};
