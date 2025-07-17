import React, { useEffect, useState } from 'react';
import './WorkflowProgress.css';

export interface WorkflowStep {
  query_id: string;
  step_number: number;
  step_name: string;
  user_friendly_name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  processing_time_ms?: number;
  technology: string;
  details: string;
  fix_applied?: string;
  progress_percentage: number;
  technical_data?: any;
}

interface WorkflowProgressProps {
  queryId: string | null;
  onComplete?: (response: any) => void;
  onError?: (error: string) => void;
  viewLayer?: 1 | 2 | 3;
}

export const WorkflowProgress: React.FC<WorkflowProgressProps> = ({
  queryId,
  onComplete,
  onError,
  viewLayer = 1
}) => {
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
    setStartTime(Date.now());

    // Establish Server-Sent Events connection
    const eventSource = new EventSource(`http://localhost:8000/api/v1/query/stream/${queryId}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.event_type === 'completion') {
          // Final response received
          setCurrentProgress(100);
          setIsStreaming(false);
          if (startTime) {
            setTotalTime(Date.now() - startTime);
          }
          if (onComplete) {
            onComplete(data.response);
          }
          eventSource.close();
        } else if (data.event_type === 'error') {
          // Error occurred
          setIsStreaming(false);
          if (onError) {
            onError(data.error);
          }
          eventSource.close();
        } else {
          // Regular step update
          const stepData = data as WorkflowStep;
          setSteps(prev => {
            const existing = prev.find(s => s.step_name === stepData.step_name);
            if (existing) {
              return prev.map(s => s.step_name === stepData.step_name ? stepData : s);
            } else {
              return [...prev, stepData];
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
      if (onError) {
        onError('Connection to server lost');
      }
      eventSource.close();
    };

    // Cleanup
    return () => {
      eventSource.close();
      setIsStreaming(false);
    };
  }, [queryId, onComplete, onError, startTime]);

  if (!queryId || steps.length === 0) {
    return null;
  }

  return (
    <div className="workflow-progress">
      <div className="workflow-header">
        <h3>🚀 Processing Your Query...</h3>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${currentProgress}%` }}
          />
        </div>
        <div className="progress-info">
          <span>{Math.round(currentProgress)}% Complete</span>
          {totalTime > 0 && <span>({(totalTime / 1000).toFixed(1)}s total)</span>}
          {isStreaming && startTime && (
            <span>({((Date.now() - startTime) / 1000).toFixed(1)}s elapsed)</span>
          )}
        </div>
      </div>

      <div className="workflow-steps">
        {steps.map((step, index) => (
          <WorkflowStepCard
            key={step.step_name}
            step={step}
            viewLayer={viewLayer}
          />
        ))}
      </div>

      {steps.length > 0 && (
        <div className="workflow-footer">
          <div className="fixes-applied">
            🔧 {steps.filter(s => s.fix_applied).length} advanced fixes applied for optimal results
          </div>
        </div>
      )}
    </div>
  );
};

interface WorkflowStepCardProps {
  step: WorkflowStep;
  viewLayer: 1 | 2 | 3;
}

const WorkflowStepCard: React.FC<WorkflowStepCardProps> = ({ step, viewLayer }) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✅';
      case 'in_progress': return '⏳';
      case 'error': return '❌';
      default: return '⏸️';
    }
  };

  const getStatusClass = (status: string) => {
    switch (status) {
      case 'completed': return 'completed';
      case 'in_progress': return 'in-progress';
      case 'error': return 'error';
      default: return 'pending';
    }
  };

  return (
    <div className={`workflow-step-card ${getStatusClass(step.status)}`}>
      <div className="step-header">
        <div className="step-title">
          <span className="status-icon">{getStatusIcon(step.status)}</span>
          <span className="step-name">
            {viewLayer === 1 ? step.user_friendly_name : `Step ${step.step_number}: ${step.step_name}`}
          </span>
          {step.fix_applied && (
            <span className="fix-badge">{step.fix_applied}</span>
          )}
        </div>
        {step.processing_time_ms && (
          <div className="step-time">
            {step.processing_time_ms < 1000
              ? `${step.processing_time_ms.toFixed(1)}ms`
              : `${(step.processing_time_ms / 1000).toFixed(1)}s`
            }
          </div>
        )}
      </div>

      {viewLayer >= 2 && (
        <div className="step-details">
          <div className="technology">
            <strong>Technology:</strong> {step.technology}
          </div>
          <div className="details">
            <strong>Result:</strong> {step.details}
          </div>
        </div>
      )}

      {viewLayer >= 3 && step.technical_data && (
        <div className="technical-data">
          <strong>Technical Details:</strong>
          <pre>{JSON.stringify(step.technical_data, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default WorkflowProgress;