import axios from "axios";
import { useState } from "react";
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
        "http://localhost:8000/api/v1/query/structured/",
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
