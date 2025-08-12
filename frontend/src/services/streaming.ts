// Streaming service for Universal RAG workflow events
import { API_CONFIG } from '../utils/api-config';

export function createWorkflowEventSource(queryId: string, onMessage: (data: Record<string, unknown>) => void, onError: (error: Event) => void): EventSource {
  const eventSource = new EventSource(`${API_CONFIG.BASE_URL}/api/v1/stream/workflow/${queryId}`);

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
