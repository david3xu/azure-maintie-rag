import React from 'react';
import { isValidQuery } from '../../utils/validators';

interface QueryFormProps {
  query: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onSubmit: (e: React.FormEvent) => void;
  loading: boolean;
  isStreaming: boolean;
}

const QueryForm: React.FC<QueryFormProps> = ({
  query,
  onChange,
  onSubmit,
  loading,
  isStreaming,
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
  </form>
);

export default QueryForm;
