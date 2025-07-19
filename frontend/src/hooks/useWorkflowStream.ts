import { useEffect, useState } from 'react';
import { API_CONFIG } from '../utils/api-config';

export const useWorkflowStream = (
  queryId: string | null,
  onComplete?: (response: any) => void,
  onError?: (error: string) => void,
  query?: string,
  domain?: string
) => {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!queryId || !query) return;

    setIsConnected(true);

    // Use real workflow endpoint with query and domain parameters
    const params = new URLSearchParams({
      query: query,
      domain: domain || 'general'
    });

    const eventSource = new EventSource(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.REAL_WORKFLOW_STREAM}/${queryId}?${params.toString()}`
    );

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Handle real workflow events
        if (data.event_type === 'workflow_completed') {
          if (onComplete) onComplete(data);
          eventSource.close();
          setIsConnected(false);
        } else if (data.event_type === 'workflow_error' || data.event_type === 'error') {
          if (onError) onError(data.error || 'Workflow failed');
          eventSource.close();
          setIsConnected(false);
        }
        // Note: Progress events are handled by the WorkflowProgress component
      } catch (error) {
        console.error('Error parsing SSE data:', error);
      }
    };

    eventSource.onerror = () => {
      if (onError) onError('Connection to server lost');
      eventSource.close();
      setIsConnected(false);
    };

    return () => {
      eventSource.close();
      setIsConnected(false);
    };
  }, [queryId, query, domain]);

  return { isConnected };
};