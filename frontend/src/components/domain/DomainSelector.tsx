import React from 'react';

interface DomainSelectorProps {
  domain: string;
  onChange: (domain: string) => void;
}

// Universal RAG system works with any markdown content from data/raw directory
const domains = [
  { value: 'general', label: 'Universal (Markdown Files)' }
];

const DomainSelector: React.FC<DomainSelectorProps> = ({ domain, onChange }) => (
  <div className="domain-selector">
    <label htmlFor="domain-select">Data Source:</label>
    <select
      id="domain-select"
      value={domain}
      onChange={e => onChange(e.target.value)}
      className="domain-dropdown"
    >
      {domains.map(d => (
        <option key={d.value} value={d.value}>{d.label}</option>
      ))}
    </select>
  </div>
);

export default DomainSelector;
