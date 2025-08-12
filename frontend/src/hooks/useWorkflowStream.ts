import { useEffect, useState } from 'react';
import { API_CONFIG } from '../utils/api-config';
import type { QueryResponse } from '../types/api';

export const useWorkflowStream = (
  queryId: string | null,
  onComplete?: (response: QueryResponse) => void,
  onError?: (error: string) => void,
  query?: string,
  domain?: string
) => {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!queryId || !query) return;

    setIsConnected(true);

    // FUNC! Connect to REAL Azure streaming endpoint - NO fake parameters
    console.log('Connecting to REAL Azure workflow stream:', queryId);
    
    const eventSource = new EventSource(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.REAL_WORKFLOW_STREAM}/${queryId}`
    );

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Handle REAL Azure workflow events - NO fake processing
        if (data.event_type === 'workflow_completed') {
          console.log('REAL Azure workflow completed:', data);
          if (onComplete) onComplete(data);
          eventSource.close();
          setIsConnected(false);
        } else if (data.event_type === 'workflow_failed' || data.event_type === 'error') {
          console.error('REAL Azure workflow FAILED:', data.error);
          if (onError) onError(data.error || 'Real Azure workflow failed');
          eventSource.close();
          setIsConnected(false);
        }
        // Note: Progress events are handled by the WorkflowProgress component
      } catch (error) {
        console.error('Error parsing SSE data:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('REAL Azure streaming connection FAILED:', error);
      if (onError) onError('REAL Azure streaming connection lost - backend or Azure services down');
      eventSource.close();
      setIsConnected(false);
    };

    return () => {
      eventSource.close();
      setIsConnected(false);
    };
  }, [queryId, query, domain, onComplete, onError]);

  return { isConnected };
};