# Add Frontend to Your Existing Azure MaintIE RAG Project

## 🎯 Step 1: Add Frontend (2 minutes)

```bash
# From your azure-maintie-rag root directory
npm create vite@latest frontend -- --template react-ts
cd frontend && npm install

# Add axios for API calls
npm install axios

cd ..  # Back to root
```

## 📄 Step 2: Root Makefile (azure-maintie-rag/Makefile)

```makefile
# Azure MaintIE RAG - Full Stack Project
# Root-level commands coordinating backend + frontend

.PHONY: help setup dev test clean health

# Default target
help:
	@echo "🔧 Azure MaintIE RAG - Full Stack Commands"
	@echo ""
	@echo "🚀 Development:"
	@echo "  make setup      - Setup both backend and frontend"
	@echo "  make dev        - Start both services (backend + frontend)"
	@echo "  make backend    - Start backend only"
	@echo "  make frontend   - Start frontend only"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test       - Run all tests"
	@echo "  make test-backend - Backend tests only"
	@echo "  make test-frontend - Frontend tests only"
	@echo ""
	@echo "🔍 Health:"
	@echo "  make health     - Check system health"
	@echo "  make check      - Verify both services running"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-up  - Start with Docker Compose"
	@echo "  make docker-down - Stop Docker services"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  make clean      - Clean all build artifacts"

# Setup everything
setup:
	@echo "🔧 Setting up backend..."
	cd backend && make setup
	@echo "🎨 Setting up frontend..."
	cd frontend && npm install
	@echo "✅ Full setup complete!"
	@echo ""
	@echo "🚀 Next: make dev"

# Start both services for development
dev:
	@echo "🚀 Starting full development environment..."
	@echo ""
	@echo "📍 Services will be available at:"
	@echo "   Backend API:  http://localhost:8000"
	@echo "   Frontend UI:  http://localhost:3000"
	@echo "   API Docs:     http://localhost:8000/docs"
	@echo ""
	@echo "Press Ctrl+C to stop both services"
	@echo ""
	make -j2 backend frontend

# Start backend only
backend:
	@echo "🔧 Starting backend service..."
	cd backend && make run

# Start frontend only
frontend:
	@echo "🎨 Starting frontend service..."
	cd frontend && npm run dev

# Test everything
test:
	@echo "🧪 Running backend tests..."
	cd backend && make test
	@echo "🧪 Running frontend tests..."
	cd frontend && npm run test:ci || echo "Frontend tests not configured yet"

test-backend:
	cd backend && make test

test-frontend:
	cd frontend && npm run test

# Health checks
health:
	@echo "🔍 Checking system health..."
	@echo ""
	@echo "Backend API:"
	@curl -s http://localhost:8000/api/v1/health || echo "❌ Backend not responding"
	@echo ""
	@echo "Frontend:"
	@curl -s http://localhost:3000 > /dev/null && echo "✅ Frontend running" || echo "❌ Frontend not running"

check:
	@echo "🔍 Quick system check..."
	@curl -s http://localhost:8000/api/v1/health > /dev/null && echo "✅ Backend OK" || echo "❌ Backend down"
	@curl -s http://localhost:3000 > /dev/null && echo "✅ Frontend OK" || echo "❌ Frontend down"

# Docker operations
docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

# Cleanup
clean:
	@echo "🧹 Cleaning backend..."
	cd backend && make clean
	@echo "🧹 Cleaning frontend..."
	cd frontend && rm -rf dist/ node_modules/.vite/
	@echo "✅ Cleanup complete"

# Quick development workflow
quick-start: setup dev
```

## 📄 Step 3: Update Frontend Integration (frontend/src/App.tsx)

```typescript
import { useState } from "react";
import axios from "axios";
import "./App.css";

// Types matching your backend API
interface QueryRequest {
  query: string;
  max_results?: number;
  include_explanations?: boolean;
  enable_safety_warnings?: boolean;
}

interface QueryResponse {
  query: string;
  generated_response: string;
  confidence_score: number;
  processing_time: number;
  safety_warnings: string[];
  sources: string[];
  citations: string[];
}

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const result = await axios.post<QueryResponse>(
        "http://localhost:8000/api/v1/query",
        {
          query,
          max_results: 10,
          include_explanations: true,
          enable_safety_warnings: true,
        } as QueryRequest
      );

      setResponse(result.data);
    } catch (err: any) {
      setError(err.response?.data?.error || "Query failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🔧 MaintIE Intelligence</h1>
        <p>Azure-Powered Maintenance Knowledge Assistant</p>
      </header>

      <main className="main">
        <form onSubmit={handleSubmit} className="query-form">
          <div className="input-group">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a maintenance question... (e.g., 'How to troubleshoot pump vibration?')"
              disabled={loading}
              className="query-input"
            />
            <button
              type="submit"
              disabled={!query.trim() || loading}
              className="submit-button"
            >
              {loading ? "Processing..." : "Ask"}
            </button>
          </div>
        </form>

        {error && (
          <div className="error">
            <h3>❌ Error</h3>
            <p>{error}</p>
            <small>
              Make sure your backend is running: <code>make backend</code>
            </small>
          </div>
        )}

        {response && (
          <div className="response">
            {/* Safety Warnings */}
            {response.safety_warnings.length > 0 && (
              <div className="safety-warnings">
                <h3>⚠️ Safety Warnings</h3>
                <ul>
                  {response.safety_warnings.map((warning, i) => (
                    <li key={i}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Response Metadata */}
            <div className="response-meta">
              <span className="confidence">
                Confidence: {(response.confidence_score * 100).toFixed(0)}%
              </span>
              <span className="timing">
                {response.processing_time.toFixed(2)}s
              </span>
              <span className="sources">{response.sources.length} sources</span>
            </div>

            {/* Main Response */}
            <div className="response-content">
              <h3>💡 Response</h3>
              <div className="response-text">{response.generated_response}</div>
            </div>

            {/* Sources */}
            {response.citations.length > 0 && (
              <div className="sources">
                <h3>📚 Sources</h3>
                <ul>
                  {response.citations.map((citation, i) => (
                    <li key={i}>{citation}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Welcome message when no query */}
        {!response && !loading && !error && (
          <div className="welcome">
            <h2>Welcome to MaintIE Intelligence</h2>
            <div className="features">
              <div className="feature">
                <h3>🔧 Troubleshooting</h3>
                <p>Get expert guidance for equipment failures</p>
              </div>
              <div className="feature">
                <h3>📋 Procedures</h3>
                <p>Access detailed maintenance procedures</p>
              </div>
              <div className="feature">
                <h3>⚠️ Safety</h3>
                <p>Automatic safety warnings and compliance</p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
```

## 📄 Step 4: Frontend Styling (frontend/src/App.css)

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f8fafc;
  color: #1e293b;
}

.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
  color: white;
  padding: 2rem;
  text-align: center;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.header h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  font-weight: 700;
}

.header p {
  opacity: 0.9;
  font-size: 1.1rem;
}

.main {
  flex: 1;
  padding: 2rem;
  max-width: 1000px;
  margin: 0 auto;
  width: 100%;
}

.query-form {
  margin-bottom: 2rem;
}

.input-group {
  display: flex;
  gap: 0.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  border-radius: 0.5rem;
  overflow: hidden;
}

.query-input {
  flex: 1;
  padding: 1.25rem 1.5rem;
  border: none;
  font-size: 1rem;
  outline: none;
}

.query-input::placeholder {
  color: #94a3b8;
}

.submit-button {
  padding: 1.25rem 2rem;
  background: #2563eb;
  color: white;
  border: none;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s;
}

.submit-button:hover:not(:disabled) {
  background: #1d4ed8;
}

.submit-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error {
  background: #fef2f2;
  border: 1px solid #fecaca;
  padding: 1.5rem;
  border-radius: 0.5rem;
  margin-bottom: 2rem;
}

.error h3 {
  color: #dc2626;
  margin-bottom: 0.5rem;
}

.error p {
  color: #b91c1c;
  margin-bottom: 0.5rem;
}

.error small {
  color: #7f1d1d;
  font-family: "Courier New", monospace;
}

.response {
  background: white;
  border-radius: 0.75rem;
  padding: 2rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  border: 1px solid #e2e8f0;
}

.safety-warnings {
  background: #fef2f2;
  border: 1px solid #fecaca;
  padding: 1.5rem;
  border-radius: 0.5rem;
  margin-bottom: 1.5rem;
}

.safety-warnings h3 {
  color: #dc2626;
  margin-bottom: 1rem;
  font-size: 1.1rem;
}

.safety-warnings ul {
  list-style: none;
}

.safety-warnings li {
  margin-bottom: 0.5rem;
  padding-left: 1rem;
  position: relative;
}

.safety-warnings li::before {
  content: "•";
  color: #dc2626;
  position: absolute;
  left: 0;
}

.response-meta {
  display: flex;
  gap: 2rem;
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #e2e8f0;
  font-size: 0.875rem;
  font-weight: 600;
}

.confidence {
  color: #059669;
}
.timing {
  color: #7c3aed;
}
.sources {
  color: #0284c7;
}

.response-content h3 {
  margin-bottom: 1rem;
  color: #1e293b;
  font-size: 1.25rem;
}

.response-text {
  line-height: 1.7;
  color: #334155;
  white-space: pre-wrap;
  margin-bottom: 1.5rem;
}

.sources {
  border-top: 1px solid #e2e8f0;
  padding-top: 1.5rem;
}

.sources h3 {
  margin-bottom: 1rem;
  color: #1e293b;
  font-size: 1.1rem;
}

.sources ul {
  list-style: none;
}

.sources li {
  margin-bottom: 0.5rem;
  padding: 0.5rem;
  background: #f8fafc;
  border-radius: 0.25rem;
  font-size: 0.875rem;
  color: #475569;
}

.welcome {
  text-align: center;
  padding: 3rem 1rem;
}

.welcome h2 {
  margin-bottom: 3rem;
  font-size: 2rem;
  color: #1e293b;
}

.features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
}

.feature {
  background: white;
  padding: 2rem;
  border-radius: 0.75rem;
  box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
  border: 1px solid #e2e8f0;
}

.feature h3 {
  margin-bottom: 1rem;
  font-size: 1.25rem;
  color: #1e293b;
}

.feature p {
  color: #64748b;
  line-height: 1.6;
}

@media (max-width: 768px) {
  .input-group {
    flex-direction: column;
  }

  .response-meta {
    flex-direction: column;
    gap: 0.5rem;
  }

  .main {
    padding: 1rem;
  }

  .header {
    padding: 1.5rem;
  }

  .header h1 {
    font-size: 2rem;
  }
}
```

## 🚀 Complete Development Workflow

```bash
# One-time setup
make setup

# Daily development (starts both services)
make dev
# Backend: http://localhost:8000
# Frontend: http://localhost:3000

# Testing
make test              # All tests
make test-backend      # Backend only

# Health check
make health            # Check if services responding

# Individual services
make backend           # Backend only
make frontend          # Frontend only

# Docker deployment
make docker-up         # Full Docker stack
```

## 📊 What You Get

✅ **Professional full-stack system** in 5 minutes
✅ **Your existing backend** unchanged and working
✅ **Modern React + TypeScript frontend**
✅ **Integrated development workflow**
✅ **Type-safe API integration**
✅ **Production-ready structure**
✅ **Good architecture** - clean separation
✅ **Good lifecycle** - unified commands
