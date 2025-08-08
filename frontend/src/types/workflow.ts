// Workflow step type for Universal RAG workflow progress
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
  technical_data?: Record<string, unknown>;
}
