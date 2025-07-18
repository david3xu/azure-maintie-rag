import { DOMAINS } from './constants';

export const isValidQuery = (query: string): boolean => {
  return query.trim().length >= 3;
};

export function isValidDomain(domain: string): boolean {
  return DOMAINS.some(d => d.value === domain);
}
