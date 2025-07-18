import { DOMAINS } from './constants';

export function isValidQuery(query: string): boolean {
  return typeof query === 'string' && query.trim().length > 0;
}

export function isValidDomain(domain: string): boolean {
  return DOMAINS.some(d => d.value === domain);
}
