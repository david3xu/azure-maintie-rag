import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => (
  <div className="App">
    <header className="App-header">
      <h1>🤖 Universal Enhanced RAG</h1>
      <p>Intelligent Knowledge Assistant with Real-Time Processing</p>
    </header>
    <main className="main-content">
      {children}
    </main>
    <footer className="app-footer">
      <p>
        🤖 Powered by Universal Smart RAG with Phase 1, 2, 3 optimizations |
        🔒 Enterprise-grade maintenance intelligence
      </p>
    </footer>
  </div>
);

export default Layout;
