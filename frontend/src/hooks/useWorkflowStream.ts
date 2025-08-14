import { useEffect, useState } from 'react';
import { API_CONFIG } from '../utils/api-config';
import { triggerWorkflowWithStreaming } from '../services/api';
import type { QueryResponse, WorkflowEvent } from '../types/api';

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

    console.log('Starting REAL Azure workflow with streaming for:', query);
    
    // First, trigger the actual Azure workflow processing
    triggerWorkflowWithStreaming(query, queryId).catch(error => {
      console.error('Failed to trigger workflow:', error);
      if (onError) onError(`Failed to start Azure workflow: ${error.message}`);
      setIsConnected(false);
      return;
    });
    
    // Then connect to streaming endpoint to show progress
    const eventSource = new EventSource(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.WORKFLOW_STREAM}/${queryId}`
    );

    eventSource.onmessage = (event) => {
      try {
        const data: WorkflowEvent = JSON.parse(event.data);
        console.log('Workflow event received:', data);

        // Handle workflow completion
        if (data.event_type === 'workflow_completed') {
          console.log('Azure workflow completed:', data);
          
          // Convert WorkflowEvent to QueryResponse format
          const response: QueryResponse = {
            query: data.query || query,
            generated_response: data.generated_response || 'Workflow completed successfully',
            confidence_score: data.confidence_score || 0.8,
            processing_time: data.processing_time || 0,
            safety_warnings: data.safety_warnings || [],
            sources: data.sources || [],
            citations: data.citations || []
          };
          
          if (onComplete) onComplete(response);
          eventSource.close();
          setIsConnected(false);
          
        } else if (data.event_type === 'workflow_failed' || data.event_type === 'error') {
          console.error('Azure workflow failed:', data.error);
          if (onError) onError(data.error || 'Azure workflow failed');
          eventSource.close();
          setIsConnected(false);
        }
        // Progress events are handled by WorkflowPanel component

      } catch (error) {
        console.error('Error parsing workflow event:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('Workflow stream connection failed:', error);
      if (onError) onError('Workflow streaming connection lost - check backend availability');
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