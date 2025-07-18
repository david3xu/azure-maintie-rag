import React from 'react';

interface DomainSelectorProps {
  domain: string;
  onChange: (domain: string) => void;
}

const domains = [
  { value: 'general', label: 'General' },
  { value: 'finance', label: 'Finance' },
  { value: 'healthcare', label: 'Healthcare' },
  { value: 'engineering', label: 'Engineering' },
  // Add more domains as needed
];

const DomainSelector: React.FC<DomainSelectorProps> = ({ domain, onChange }) => (
  <div className="domain-selector">
    <label htmlFor="domain-select">Domain:</label>
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
