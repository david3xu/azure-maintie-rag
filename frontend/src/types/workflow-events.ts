export interface WorkflowEventType {
  COMPLETED: 'workflow_completed';
  FAILED: 'workflow_failed';
  ERROR: 'error';
  PROGRESS: 'progress';
  HEARTBEAT: 'heartbeat';
}

export const WORKFLOW_EVENTS: WorkflowEventType = {
  COMPLETED: 'workflow_completed',
  FAILED: 'workflow_failed',
  ERROR: 'error',
  PROGRESS: 'progress',
  HEARTBEAT: 'heartbeat'
};