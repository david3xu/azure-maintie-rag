import { useEffect, useState } from 'react';
import type { WorkflowStep } from '../types/workflow';
import { WORKFLOW_EVENTS } from '../types/workflow-events';
import { API_CONFIG } from '../utils/api-config';

export const useWorkflowStream = (
  queryId: string | null,
  onComplete?: (response: any) => void,
  onError?: (error: string) => void
) => {
  const [steps, setSteps] = useState<WorkflowStep[]>([]);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [totalTime, setTotalTime] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);

  useEffect(() => {
    if (!queryId) return;

    setIsStreaming(true);
    setSteps([]);
    setCurrentProgress(0);
    const connectionStartTime = Date.now();
    setStartTime(connectionStartTime);

    const eventSource = new EventSource(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.STREAM}/${queryId}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.event_type === WORKFLOW_EVENTS.COMPLETED) {
          setCurrentProgress(100);
          setIsStreaming(false);
          setTotalTime(Date.now() - connectionStartTime);
          if (onComplete) onComplete(data);
          eventSource.close();

        } else if (data.event_type === WORKFLOW_EVENTS.FAILED || data.event_type === WORKFLOW_EVENTS.ERROR) {
          setIsStreaming(false);
          if (onError) onError(data.error || data.message || 'Workflow failed');
          eventSource.close();

        } else if (data.event_type === WORKFLOW_EVENTS.PROGRESS) {
          const stepData = data as WorkflowStep;
          setSteps(prev => {
            const existing = prev.find(s => s.step_number === stepData.step_number);
            if (existing) {
              return prev.map(s => s.step_number === stepData.step_number ? stepData : s);
            } else {
              const newSteps = [...prev, stepData];
              return newSteps.sort((a, b) => a.step_number - b.step_number);
            }
          });
          setCurrentProgress(stepData.progress_percentage);
        }
      } catch (error) {
        console.error('Error parsing SSE data:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      setIsStreaming(false);
      if (currentProgress < 100 && onError) {
        onError('Connection to server lost');
      }
      eventSource.close();
    };

    return () => {
      eventSource.close();
      setIsStreaming(false);
    };
  }, [queryId]);

  return {
    steps,
    currentProgress,
    isStreaming,
    totalTime,
    startTime
  };
};