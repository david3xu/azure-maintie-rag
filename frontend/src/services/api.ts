import axios from 'axios';
import type { QueryResponse, UniversalQueryRequest } from '../types/api';
import { API_CONFIG } from '../utils/api-config';

export async function postUniversalQuery(
  query: string,
  domain: string = 'general'
): Promise<QueryResponse> {
  const response = await axios.post<QueryResponse>(
    `${API_CONFIG.BASE_URL}/query/universal`,
    {
      query: query.trim(),
      domain: domain,
      max_results: 10,
      include_explanations: true,
      enable_safety_warnings: true,
    } as UniversalQueryRequest
  );
  return response.data;
}

export async function getWorkflowSummary(queryId: string): Promise<any> {
  const response = await axios.get(`${API_CONFIG.BASE_URL}/workflow/${queryId}/summary`);
  return response.data;
}
