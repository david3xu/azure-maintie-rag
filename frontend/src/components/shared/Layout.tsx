import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
  isDarkMode?: boolean;
  onThemeToggle?: () => void;
  // Add new props for header controls
  domain?: string;
  onDomainChange?: (domain: string) => void;
  showWorkflow?: boolean;
  onWorkflowToggle?: (checked: boolean) => void;
  viewLayer?: 1 | 2 | 3;
  onViewLayerChange?: (layer: 1 | 2 | 3) => void;
  loading?: boolean;
  isStreaming?: boolean;
}

const Layout: React.FC<LayoutProps> = ({
  children,
  isDarkMode,
  onThemeToggle,
  domain,
  onDomainChange,
  showWorkflow,
  onWorkflowToggle,
  viewLayer,
  onViewLayerChange,
  loading,
  isStreaming
}) => (
  <div className="App">
    <header className="App-header">
      <div className="header-content">
        <div className="header-main">
          <h1>🤖 Universal Enhanced RAG</h1>
          <p>Intelligent Knowledge Assistant with Real-Time Processing</p>
        </div>
        <div className="header-controls">
          {/* Domain Selector */}
          {onDomainChange && (
            <div className="header-control-group">
              <label htmlFor="header-domain-select" className="header-label">Domain:</label>
              <select
                id="header-domain-select"
                value={domain}
                onChange={e => onDomainChange(e.target.value)}
                className="header-select"
                disabled={loading || isStreaming}
              >
                <option value="general">General</option>
                <option value="general">General</option>
                <option value="engineering">Engineering</option>
              </select>
            </div>
          )}

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

          {/* View Layer Selector - only show when workflow is enabled */}
          {showWorkflow && onViewLayerChange && (
            <div className="header-control-group">
              <label htmlFor="header-view-layer" className="header-label">Detail:</label>
              <select
                id="header-view-layer"
                value={viewLayer}
                onChange={e => onViewLayerChange(Number(e.target.value) as 1 | 2 | 3)}
                disabled={loading || isStreaming}
                className="header-select header-select-compact"
              >
                <option value={1}>🔍 User-Friendly</option>
                <option value={2}>🔧 Technical</option>
                <option value={3}>🔬 Diagnostics</option>
              </select>
            </div>
          )}

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
        🤖 Powered by Universal Smart RAG with Phase 1, 2, 3 optimizations |
        🔒 Enterprise-grade universal intelligence
      </p>
    </footer>
  </div>
);

export default Layout;
