import { useEffect, useState } from 'react';
import { API_CONFIG } from '../utils/api-config';

export const useWorkflowStream = (
  queryId: string | null,
  onComplete?: (response: any) => void,
  onError?: (error: string) => void
) => {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!queryId) return;

    setIsConnected(true);
    const eventSource = new EventSource(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.STREAM}/${queryId}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.event_type === 'workflow_completed') {
          if (onComplete) onComplete(data);
          eventSource.close();
          setIsConnected(false);
        } else if (data.event_type === 'workflow_failed' || data.event_type === 'error') {
          if (onError) onError(data.error || 'Workflow failed');
          eventSource.close();
          setIsConnected(false);
        }
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
  }, [queryId]);

  return { isConnected };
};