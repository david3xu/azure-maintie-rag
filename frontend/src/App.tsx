import axios from "axios";
import { useState } from "react";
import "./App.css";
import WorkflowProgress from "./components/WorkflowProgress";

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

interface StreamingQueryResponse {
  query_id: string;
  status: string;
  stream_url: string;
}

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // New state for workflow progress
  const [isStreaming, setIsStreaming] = useState(false);
  const [queryId, setQueryId] = useState<string | null>(null);
  const [viewLayer, setViewLayer] = useState<1 | 2 | 3>(1);
  const [showWorkflow, setShowWorkflow] = useState(true);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);
    setQueryId(null);

    if (showWorkflow) {
      // Use streaming endpoint for workflow visibility
      try {
        const streamingResponse = await axios.post<StreamingQueryResponse>(
          "http://localhost:8000/api/v1/query/streaming",
          {
            query,
            max_results: 10,
            include_explanations: true,
            enable_safety_warnings: true,
          } as QueryRequest
        );

        setQueryId(streamingResponse.data.query_id);
        setIsStreaming(true);
        setLoading(false); // Let WorkflowProgress handle the loading state

      } catch (err: any) {
        setError(err.response?.data?.error || "Failed to start streaming query");
        setLoading(false);
        setIsStreaming(false);
      }
    } else {
      // Use original structured endpoint for simple response
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
    }
  };

  const handleWorkflowComplete = (workflowResponse: QueryResponse) => {
    setResponse(workflowResponse);
    setIsStreaming(false);
    setLoading(false);
  };

  const handleWorkflowError = (errorMessage: string) => {
    setError(errorMessage);
    setIsStreaming(false);
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔧 MaintIE Enhanced RAG</h1>
        <p>Intelligent Maintenance Assistance with Real-Time Processing</p>
      </header>

      <main className="main-content">
        <form onSubmit={handleSubmit} className="query-form">
          <div className="input-group">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask your maintenance question... (e.g., 'What causes pump bearing failure?')"
              className="query-input"
              rows={3}
              disabled={loading || isStreaming}
            />
            <button
              type="submit"
              disabled={!query.trim() || loading || isStreaming}
              className="submit-button"
            >
              {loading || isStreaming ? "Processing..." : "Ask Question"}
            </button>
          </div>

          {/* Workflow Settings */}
          <div className="workflow-settings">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={showWorkflow}
                onChange={(e) => setShowWorkflow(e.target.checked)}
                disabled={loading || isStreaming}
              />
              Show real-time processing workflow
            </label>

            {showWorkflow && (
              <div className="view-controls">
                <label>Detail Level:</label>
                <select
                  value={viewLayer}
                  onChange={(e) => setViewLayer(Number(e.target.value) as 1 | 2 | 3)}
                  disabled={loading || isStreaming}
                  className="view-selector"
                >
                  <option value={1}>🔍 User-Friendly</option>
                  <option value={2}>🔧 Technical Details</option>
                  <option value={3}>🔬 System Diagnostics</option>
                </select>
              </div>
            )}
          </div>
        </form>

        {/* Error Display */}
        {error && (
          <div className="error-message">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {/* Loading State (for non-streaming queries) */}
        {loading && !isStreaming && (
          <div className="loading-message">
            <div className="loading-spinner"></div>
            <p>Processing your maintenance query...</p>
          </div>
        )}

        {/* Workflow Progress (for streaming queries) */}
        {showWorkflow && queryId && (
          <WorkflowProgress
            queryId={queryId}
            onComplete={handleWorkflowComplete}
            onError={handleWorkflowError}
            viewLayer={viewLayer}
          />
        )}

        {/* Response Display */}
        {response && (
          <div className="response-section">
            <div className="response-header">
              <h2>📝 Response</h2>
              <div className="response-metadata">
                <span className="confidence">
                  🎯 Confidence: {Math.round(response.confidence_score * 100)}%
                </span>
                <span className="processing-time">
                  ⏱️ Time: {response.processing_time.toFixed(1)}s
                </span>
              </div>
            </div>

            <div className="response-content">
              <div className="generated-response">
                <h3>Answer:</h3>
                <div className="response-text">
                  {response.generated_response.split('\n').map((line, index) => (
                    <p key={index}>{line}</p>
                  ))}
                </div>
              </div>

              {/* Safety Warnings */}
              {response.safety_warnings && response.safety_warnings.length > 0 && (
                <div className="safety-warnings">
                  <h4>⚠️ Safety Warnings:</h4>
                  <ul>
                    {response.safety_warnings.map((warning, index) => (
                      <li key={index}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Sources */}
              {response.sources && response.sources.length > 0 && (
                <div className="sources">
                  <h4>📚 Sources:</h4>
                  <ul>
                    {response.sources.map((source, index) => (
                      <li key={index}>{source}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Citations */}
              {response.citations && response.citations.length > 0 && (
                <div className="citations">
                  <h4>📖 Citations:</h4>
                  <ul>
                    {response.citations.map((citation, index) => (
                      <li key={index}>{citation}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Workflow Summary (if streaming was used) */}
            {showWorkflow && queryId && (
              <div className="workflow-summary">
                <h4>🚀 Processing Summary:</h4>
                <p>
                  Your query was processed through a sophisticated 5-step Universal Smart RAG pipeline
                  with 3 advanced optimizations applied for enhanced accuracy and performance.
                </p>
              </div>
            )}
          </div>
        )}

        {/* Help Text */}
        {!response && !loading && !isStreaming && (
          <div className="help-text">
            <h3>💡 How to Use</h3>
            <p>
              Ask any maintenance-related question and watch our AI system process it step-by-step.
              Enable "Show real-time processing workflow" to see the sophisticated RAG pipeline in action.
            </p>
            <div className="example-queries">
              <h4>Example Questions:</h4>
              <ul>
                <li>"What causes pump bearing failure and how can I prevent it?"</li>
                <li>"How do I troubleshoot centrifugal pump vibration issues?"</li>
                <li>"What are the safety procedures for motor maintenance?"</li>
                <li>"How often should I replace hydraulic filters?"</li>
              </ul>
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          🤖 Powered by Universal Smart RAG with Phase 1, 2, 3 optimizations |
          🔒 Enterprise-grade maintenance intelligence
        </p>
      </footer>
    </div>
  );
}

export default App;
