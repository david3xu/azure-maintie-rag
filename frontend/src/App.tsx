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

        const receivedQueryId = streamingResponse.data.query_id;
        setQueryId(receivedQueryId);
        setIsStreaming(true);
        setLoading(false);

        // Add delay before starting to poll to give backend time to register workflow
        setTimeout(() => {
          // Start polling with better error handling
          pollWorkflowStatus(receivedQueryId);
        }, 1000); // 1 second delay

      } catch (err: any) {
        console.error("Streaming query failed:", err);
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
        console.error("Structured query failed:", err);
        setError(err.response?.data?.error || "Failed to process query");
      } finally {
        setLoading(false);
      }
    }
  };

  // Add new polling function with exponential backoff
  const pollWorkflowStatus = async (queryId: string, maxRetries: number = 30) => {
    let retryCount = 0;
    let backoffDelay = 1000; // Start with 1 second

    const poll = async (): Promise<void> => {
      try {
        // Try to access the streaming endpoint
        const eventSource = new EventSource(`http://localhost:8000/api/v1/query/stream/${queryId}`);

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log("Received workflow event:", data);

            if (data.event_type === "workflow_completed") {
              setIsStreaming(false);
              eventSource.close();

              // Fetch final results if needed
              fetchFinalResults(queryId);
            } else if (data.event_type === "error" || data.event_type === "workflow_failed") {
              setError(data.message || "Workflow failed");
              setIsStreaming(false);
              eventSource.close();
            }
          } catch (parseError) {
            console.error("Failed to parse event data:", parseError);
          }
        };

        eventSource.onerror = (error) => {
          console.error("EventSource error:", error);
          eventSource.close();

          // Retry with exponential backoff
          if (retryCount < maxRetries) {
            retryCount++;
            console.log(`Retrying in ${backoffDelay}ms (attempt ${retryCount}/${maxRetries})`);

            setTimeout(() => {
              backoffDelay = Math.min(backoffDelay * 1.5, 10000); // Max 10 seconds
              poll();
            }, backoffDelay);
          } else {
            setError(`Failed to connect to workflow stream after ${maxRetries} attempts`);
            setIsStreaming(false);
          }
        };

      } catch (error) {
        console.error("Failed to start polling:", error);
        setError("Failed to start workflow monitoring");
        setIsStreaming(false);
      }
    };

    // Start polling
    poll();
  };

  // Add function to fetch final results
  const fetchFinalResults = async (queryId: string) => {
    try {
      const response = await axios.get(`http://localhost:8000/api/v1/workflow/${queryId}/summary`);

      if (response.data.success) {
        const summary = response.data.workflow_summary;

        // Convert workflow summary to QueryResponse format for display
        const queryResponse: QueryResponse = {
          query: summary.query_text || query,
          generated_response: summary.final_response || "Workflow completed successfully",
          confidence_score: summary.confidence_score || 0.9,
          processing_time: summary.total_processing_time || 0,
          safety_warnings: summary.safety_warnings || [],
          sources: summary.sources || [],
          citations: summary.citations || []
        };

        setResponse(queryResponse);
      }
    } catch (error) {
      console.error("Failed to fetch final results:", error);
      setError("Workflow completed but failed to fetch results");
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
