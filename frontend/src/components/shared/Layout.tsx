import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
  isDarkMode?: boolean;
  onThemeToggle?: () => void;
  // Add new props for header controls
  // Domain props removed - violates zero hardcoded domain bias rule
  showWorkflow?: boolean;
  onWorkflowToggle?: (checked: boolean) => void;
  // View layer props removed - fake feature not in backend
  loading?: boolean;
  isStreaming?: boolean;
}

const Layout: React.FC<LayoutProps> = ({
  children,
  isDarkMode,
  onThemeToggle,
  // Domain parameters removed - zero hardcoded domain bias
  showWorkflow,
  onWorkflowToggle,
  // View layer parameters removed - fake feature not in backend
  loading,
  isStreaming
}) => (
  <div className="App">
    <header className="App-header">
      <div className="header-content">
        <div className="header-main">
          <h1>🤖 Azure Universal RAG</h1>
          <p>Knowledge Search with Real Azure AI Services</p>
        </div>
        <div className="header-controls">
          {/* Domain Selector */}
          {/* Domain selection removed - violates zero hardcoded domain bias rule */}

          {/* Workflow Toggle */}
          {onWorkflowToggle && (
            <div className="header-control-group">
              <label className="header-checkbox-label">
                <input
                  type="checkbox"
                  checked={showWorkflow}
                  onChange={e => onWorkflowToggle(e.target.checked)}
                  disabled={loading || isStreaming}
                  className="header-checkbox"
                />
                <span className="header-checkbox-text">Real-time workflow</span>
              </label>
            </div>
          )}

          {/* View Layer Selector removed - fake feature not in backend */}

          {/* Theme Toggle */}
          {onThemeToggle && (
            <button
              className="theme-toggle"
              onClick={onThemeToggle}
              title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
              {isDarkMode ? "☀️" : "🌙"}
            </button>
          )}
        </div>
      </div>
    </header>
    <main className="main-content">
      {children}
    </main>
    <footer className="app-footer">
      <p>
        🤖 Powered by Azure Universal RAG System |
        🔒 Real Azure services with PydanticAI agents
      </p>
    </footer>
  </div>
);

export default Layout;
