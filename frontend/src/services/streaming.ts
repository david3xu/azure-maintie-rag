// Streaming service for Universal RAG workflow events
export function createWorkflowEventSource(queryId: string, onMessage: (data: Record<string, unknown>) => void, onError: (error: Event) => void): EventSource {
  const eventSource = new EventSource(`http://localhost:8000/api/v1/query/stream/${queryId}`);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (err) {
      console.error('Error parsing SSE data:', err);
    }
  };

  eventSource.onerror = (error) => {
    onError(error);
    eventSource.close();
  };

  return eventSource;
}
