import React from 'react';
import type { WorkflowStep } from '../../types/workflow';

interface WorkflowStepCardProps {
  step: WorkflowStep;
  viewLayer: 1 | 2 | 3;
}

export const WorkflowStepCard: React.FC<WorkflowStepCardProps> = ({ step, viewLayer }) => {
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
            {viewLayer === 1 ?
              step.user_friendly_name :
              `Step ${step.step_number}: ${step.step_name}`
            }
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
