import React, { useEffect, useState } from "react";
import "./App.css";
import ChatHistory from "./components/chat/ChatHistory";
import QueryForm from "./components/chat/QueryForm";
import Layout from "./components/shared/Layout";
import WorkflowPanel from "./components/workflow/WorkflowPanel";
import { useChat } from "./hooks/useChat";
import { useUniversalRAG } from "./hooks/useUniversalRAG";
import { useWorkflow } from "./hooks/useWorkflow";
import { useWorkflowStream } from "./hooks/useWorkflowStream";
import type { ChatMessage } from "./types/chat";

function App() {
  // Chat and workflow state from hooks
  const { chatHistory, setChatHistory, currentChatId, setCurrentChatId } = useChat();
  const { queryId, setQueryId } = useWorkflow();
  const [query, setQuery] = useState("");
  const [domain, setDomain] = useState("general");
  const [showWorkflow, setShowWorkflow] = useState(false);
  const [viewLayer, setViewLayer] = useState<1 | 2 | 3>(1);

  // Dark mode state management
  const [isDarkMode, setIsDarkMode] = useState(() => {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  // Apply dark mode class to document
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark-mode');
    } else {
      document.documentElement.classList.remove('dark-mode');
    }
  }, [isDarkMode]);

  // Universal RAG orchestration
  const { loading, error, runUniversalRAG } = useUniversalRAG();

  // Workflow streaming (SSE/polling)
  const [workflowError, setWorkflowError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  useWorkflowStream(
    showWorkflow ? queryId : null,
    handleWorkflowComplete,
    handleWorkflowError,
    query,
    domain
  );

  // Handle workflow completion
  function handleWorkflowComplete(response: any) {
    setIsStreaming(false);
    if (currentChatId) {
      setChatHistory(prev =>
        prev.map(msg =>
          msg.id === currentChatId
            ? { ...msg, response, isStreaming: false }
            : msg
        )
      );
    }
  }

  // Handle workflow error
  function handleWorkflowError(errorMessage: string) {
    setWorkflowError(errorMessage);
    setIsStreaming(false);
    if (currentChatId) {
      setChatHistory(prev =>
        prev.map(msg =>
          msg.id === currentChatId
            ? { ...msg, error: errorMessage, isStreaming: false }
            : msg
        )
      );
    }
  }

  // Handle query submission
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
    setQueryId(null);
    setWorkflowError(null);

    if (showWorkflow) {
      setIsStreaming(true);
      setQueryId(chatId); // Use chatId as queryId for streaming
    } else {
      try {
        const queryResponse = await runUniversalRAG({ query: query.trim(), domain });
        setChatHistory(prev =>
          prev.map(msg =>
            msg.id === chatId
              ? { ...msg, response: queryResponse, isStreaming: false }
              : msg
          )
        );
      } catch (err: any) {
        setChatHistory(prev =>
          prev.map(msg =>
            msg.id === chatId
              ? { ...msg, error: error || err.message, isStreaming: false }
              : msg
          )
        );
      }
    }
    setQuery("");
  };

  return (
    <Layout
      isDarkMode={isDarkMode}
      onThemeToggle={() => setIsDarkMode(!isDarkMode)}
      domain={domain}
      onDomainChange={setDomain}
      showWorkflow={showWorkflow}
      onWorkflowToggle={setShowWorkflow}
      viewLayer={viewLayer}
      onViewLayerChange={setViewLayer}
      loading={loading || isStreaming}
      isStreaming={isStreaming}
    >
      <QueryForm
        query={query}
        onChange={e => setQuery(e.target.value)}
        onSubmit={handleSubmit}
        loading={loading || isStreaming}
        isStreaming={isStreaming}
      />
      <div className="split-layout">
        <div className="chat-panel">
          <ChatHistory
            messages={chatHistory}
            currentChatId={currentChatId}
          />
        </div>
        <WorkflowPanel
          showWorkflow={showWorkflow}
          queryId={queryId}
          viewLayer={viewLayer}
          onComplete={handleWorkflowComplete}
          onError={handleWorkflowError}
        />
      </div>
      {(error || workflowError) && (
        <div className="error-message">
          <h3>❌ Error</h3>
          <p>{error || workflowError}</p>
        </div>
      )}
    </Layout>
  );
}

export default App;
