import React from 'react';
import WorkflowProgress from './WorkflowProgress';
import type { QueryResponse } from '../../types/api';

interface WorkflowPanelProps {
  showWorkflow: boolean;
  queryId: string | null;
  // viewLayer removed - fake feature not in backend
  onComplete: (response: QueryResponse) => void;
  onError: (error: string) => void;
}

const WorkflowPanel: React.FC<WorkflowPanelProps> = ({
  showWorkflow,
  queryId,
  // viewLayer removed - fake feature not in backend
  onComplete,
  onError,
}) => (
  <div className="workflow-panel">
    {showWorkflow && queryId ? (
      <WorkflowProgress
        queryId={queryId}
        onComplete={onComplete}
        onError={onError}
        // viewLayer removed - fake feature not in backend
      />
    ) : (
      <div style={{ padding: '2rem', textAlign: 'center', color: '#6c757d' }}>
        <p>Enable "Show real-time processing workflow" to see step-by-step processing.</p>
      </div>
    )}
  </div>
);

export default WorkflowPanel;