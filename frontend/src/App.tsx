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

interface ChatMessage {
  id: string;
  query: string;
  response: QueryResponse | null;
  timestamp: Date;
  isStreaming: boolean;
  queryId: string | null;
  error: string | null;
}

function App() {
  // New state for chat history
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
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

    // Create new chat message
    const chatId = Date.now().toString();
    const newChatMessage: ChatMessage = {
      id: chatId,
      query: query.trim(),
      response: null,
      timestamp: new Date(),
      isStreaming: showWorkflow,
      queryId: null,
      error: null
    };

    setChatHistory(prev => [...prev, newChatMessage]);
    setCurrentChatId(chatId);
    setLoading(true);
    setError(null);
    setQueryId(null);

    if (showWorkflow) {
      try {
        const streamingResponse = await axios.post<StreamingQueryResponse>(
          "http://localhost:8000/api/v1/query/streaming",
          {
            query: query.trim(),
            max_results: 10,
            include_explanations: true,
            enable_safety_warnings: true,
          } as QueryRequest
        );
        const receivedQueryId = streamingResponse.data.query_id;
        setQueryId(receivedQueryId);
        setIsStreaming(true);
        setLoading(false);
        setChatHistory(prev =>
          prev.map(msg =>
            msg.id === chatId
              ? { ...msg, queryId: receivedQueryId }
              : msg
          )
        );
        setTimeout(() => {
          pollWorkflowStatus(receivedQueryId);
        }, 1000);
      } catch (err: any) {
        console.error("Streaming query failed:", err);
        const errorMessage = err.response?.data?.error || "Failed to start streaming query";
        setError(errorMessage);
        setChatHistory(prev =>
          prev.map(msg =>
            msg.id === chatId
              ? { ...msg, error: errorMessage, isStreaming: false }
              : msg
          )
        );
        setLoading(false);
        setIsStreaming(false);
      }
    } else {
      try {
        const result = await axios.post<QueryResponse>(
          "http://localhost:8000/api/v1/query/structured/",
          {
            query: query.trim(),
            max_results: 10,
            include_explanations: true,
            enable_safety_warnings: true,
          } as QueryRequest
        );
        setChatHistory(prev =>
          prev.map(msg =>
            msg.id === chatId
              ? { ...msg, response: result.data, isStreaming: false }
              : msg
          )
        );
      } catch (err: any) {
        console.error("Structured query failed:", err);
        const errorMessage = err.response?.data?.error || "Failed to process query";
        setError(errorMessage);
        setChatHistory(prev =>
          prev.map(msg =>
            msg.id === chatId
              ? { ...msg, error: errorMessage, isStreaming: false }
              : msg
          )
        );
      } finally {
        setLoading(false);
      }
    }
    setQuery("");
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
          // FIX: Check both final_response and generated_response fields
          generated_response: summary.final_response || summary.generated_response || "Workflow completed successfully",
          confidence_score: summary.confidence_score || 0.9,
          processing_time: summary.total_time_ms !== undefined ? (summary.total_time_ms / 1000) : (summary.total_processing_time || 0),
          safety_warnings: summary.safety_warnings || [],
          sources: summary.sources || [],
          citations: summary.citations || []
        };

        setChatHistory(prev =>
          prev.map(msg =>
            msg.id === currentChatId
              ? { ...msg, response: queryResponse, isStreaming: false }
              : msg
          )
        );
      }
    } catch (error) {
      console.error("Failed to fetch final results:", error);
      setError("Workflow completed but failed to fetch results");
    }
  };

  const handleWorkflowComplete = (workflowResponse: QueryResponse) => {
    setChatHistory(prev =>
      prev.map(msg =>
        msg.id === currentChatId
          ? { ...msg, response: workflowResponse, isStreaming: false }
          : msg
      )
    );
  };

  const handleWorkflowError = (errorMessage: string) => {
    setError(errorMessage);
    setIsStreaming(false);
    setLoading(false);
    if (currentChatId) {
      setChatHistory(prev =>
        prev.map(msg =>
          msg.id === currentChatId
            ? { ...msg, error: errorMessage, isStreaming: false }
            : msg
        )
      );
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 Universal Enhanced RAG</h1>
        <p>Intelligent Knowledge Assistant with Real-Time Processing</p>
      </header>
      <main className="main-content">
        {/* Query Form - Always visible */}
        <form onSubmit={handleSubmit} className="query-form">
          <div className="input-group">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask any question about your domain... (e.g., 'What causes pump bearing failure?')"
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
        {/* Global Error Display */}
        {error && !currentChatId && (
          <div className="error-message">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}
        {/* Split Layout - Show only when there's chat history */}
        {chatHistory.length > 0 ? (
          <div className="split-layout">
            {/* Left Panel - Chat History */}
            <div className="chat-panel">
              <div className="chat-history">
                {chatHistory.map((message) => (
                  <div
                    key={message.id}
                    className={`chat-message ${message.id === currentChatId ? 'current' : ''}`}
                  >
                    <div className="chat-message-query">
                      Q: {message.query}
                    </div>
                    <div className="chat-message-timestamp">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                    {message.error ? (
                      <div className="chat-message-error">
                        ❌ Error: {message.error}
                      </div>
                    ) : message.isStreaming && !message.response ? (
                      <div className="chat-message-loading">
                        🔄 Processing...
                      </div>
                    ) : message.response ? (
                      <div>
                        <div className="chat-message-response">
                          📝 {typeof message.response.generated_response === 'string'
                            ? message.response.generated_response
                            : JSON.stringify(message.response.generated_response)}
                        </div>
                        <div className="chat-message-metadata">
                          <span className="confidence-badge">
                            🎯 {Math.round(message.response.confidence_score * 100)}%
                          </span>
                          <span className="processing-time-badge">
                            ⏱️ {(message.response.processing_time || 0).toFixed(1)}s
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="chat-message-loading">
                        ⏳ Waiting for response...
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            {/* Right Panel - Workflow Progress */}
            <div className="workflow-panel">
              {showWorkflow && queryId ? (
                <WorkflowProgress
                  queryId={queryId}
                  onComplete={handleWorkflowComplete}
                  onError={handleWorkflowError}
                  viewLayer={viewLayer}
                />
              ) : (
                <div style={{ padding: '2rem', textAlign: 'center', color: '#6c757d' }}>
                  <p>Enable "Show real-time processing workflow" to see step-by-step processing.</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Welcome Screen - Show when no chat history */
          <div className="help-text">
            <h3>🚀 Welcome to Universal Enhanced RAG</h3>
            <p>
              Ask any question about your domain and get intelligent responses with real-time processing insights.
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
