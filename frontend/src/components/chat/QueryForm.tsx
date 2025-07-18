import React from 'react';
import { isValidQuery } from '../../utils/validators';

interface QueryFormProps {
  query: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onSubmit: (e: React.FormEvent) => void;
  loading: boolean;
  isStreaming: boolean;
  showWorkflow: boolean;
  setShowWorkflow: (checked: boolean) => void;
  viewLayer: 1 | 2 | 3;
  setViewLayer: (layer: 1 | 2 | 3) => void;
}

const QueryForm: React.FC<QueryFormProps> = ({
  query,
  onChange,
  onSubmit,
  loading,
  isStreaming,
  showWorkflow,
  setShowWorkflow,
  viewLayer,
  setViewLayer,
}) => (
  <form onSubmit={onSubmit} className="query-form">
    <div className="input-group">
      <textarea
        value={query}
        onChange={onChange}
        placeholder="Ask any question about your domain... (e.g., 'What are common issues and how to prevent them?')"
        className="query-input"
        rows={3}
        disabled={loading || isStreaming}
      />
      <button
        type="submit"
        disabled={!isValidQuery(query) || loading || isStreaming}
        className="submit-button"
      >
        {loading || isStreaming ? 'Processing...' : 'Ask Question'}
      </button>
    </div>
    {!isValidQuery(query) && query.length > 0 && (
      <div className="validation-message" style={{ color: 'red', marginTop: 4 }}>
        Please enter a valid question.
      </div>
    )}
    <div className="workflow-settings">
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={showWorkflow}
          onChange={e => setShowWorkflow(e.target.checked)}
          disabled={loading || isStreaming}
        />
        Show real-time processing workflow
      </label>
      {showWorkflow && (
        <div className="view-controls">
          <label>Detail Level:</label>
          <select
            value={viewLayer}
            onChange={e => setViewLayer(Number(e.target.value) as 1 | 2 | 3)}
            disabled={loading || isStreaming}
            className="view-selector"
          >
            <option value={1}>🔍 User-Friendly</option>
            <option value={2}>🔧 Technical Details</option>
            <option value={3}>🔬 System Diagnostics</option>
          </select>
        </div>
      )}
    </div>
  </form>
);

export default QueryForm;
