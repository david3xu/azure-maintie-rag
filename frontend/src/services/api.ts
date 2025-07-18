import axios from 'axios';
import type { QueryResponse, UniversalQueryRequest, UniversalQueryResponse } from '../types/api';
import { API_CONFIG } from '../utils/api-config';

export async function postUniversalQuery(
  query: string,
  domain: string = 'general'
): Promise<QueryResponse> {
  const response = await axios.post<UniversalQueryResponse>(
    `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.UNIVERSAL_QUERY}`,
    {
      query: query.trim(),
      domain: domain,
      max_results: 10,
      include_explanations: true,
      enable_safety_warnings: true,
    } as UniversalQueryRequest
  );
  const backendData = response.data;
  return {
    query: backendData.query,
    generated_response: backendData.generated_response.answer || 'No response generated',
    confidence_score: backendData.generated_response.confidence || 0,
    processing_time: backendData.processing_time,
    safety_warnings: [],
    sources: backendData.search_results.map((r: { source: string }) => r.source),
    citations: backendData.search_results.map((r: { content: string }) => r.content.substring(0, 100) + '...')
  };
}

export async function getWorkflowSummary(queryId: string): Promise<any> {
  const response = await axios.get(`${API_CONFIG.BASE_URL}/workflow/${queryId}/summary`);
  return response.data;
}
