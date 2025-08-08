import axios from 'axios';
import type { QueryResponse, UniversalQueryRequest, UniversalQueryResponse } from '../types/api';
import { API_CONFIG } from '../utils/api-config';

export async function postUniversalQuery(
  query: string,
  domain: string = 'general'
): Promise<QueryResponse> {
  try {
    console.log('Making API request to:', `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.UNIVERSAL_QUERY}`);
    console.log('Request payload:', { query: query.trim(), domain, max_results: 10, include_explanations: true, enable_safety_warnings: true });

    const response = await axios.post<UniversalQueryResponse>(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.UNIVERSAL_QUERY}`,
      {
        query: query.trim(),
        domain: domain,
        max_results: 3,
        include_explanations: true,
        enable_safety_warnings: true,
      } as UniversalQueryRequest,
      {
        timeout: 60000, // 60 second timeout
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    console.log('API response received:', response.data);
    console.log('Generated response type:', typeof response.data.generated_response);
    console.log('Generated response length:', response.data.generated_response ? (typeof response.data.generated_response === 'string' ? response.data.generated_response.length : 'object') : 0);
    console.log('Search results count:', response.data.search_results ? response.data.search_results.length : 0);
    const backendData = response.data;

    // Handle the actual backend response structure
    let generatedResponse = 'No response generated';
    let confidenceScore = 0;

    if (backendData.generated_response) {
      // Check if it's a string (direct response)
      if (typeof backendData.generated_response === 'string') {
        generatedResponse = backendData.generated_response;
      }
      // Check if it's an object with answer property
      else if (backendData.generated_response.answer) {
        generatedResponse = backendData.generated_response.answer;
        confidenceScore = backendData.generated_response.confidence || 0;
      }
      // Check if it's an object with content property
      else if (backendData.generated_response.content) {
        generatedResponse = backendData.generated_response.content;
      }
      // Check if it's an object with response property
      else if (backendData.generated_response.response) {
        generatedResponse = backendData.generated_response.response;
      }
      // Fallback: try to stringify the object
      else {
        generatedResponse = JSON.stringify(backendData.generated_response);
      }
    }

    return {
      query: backendData.query,
      generated_response: generatedResponse,
      confidence_score: confidenceScore,
      processing_time: backendData.processing_time,
      safety_warnings: backendData.safety_warnings || [],
      sources: backendData.search_results?.map((r: Record<string, unknown>) => (r.source as string) || ((r.metadata as Record<string, unknown>)?.source as string) || 'Unknown') || [],
      citations: backendData.search_results?.map((r: Record<string, unknown>) => {
        const content = (r.content as string) || '';
        return content.substring(0, 100) + (content.length > 100 ? '...' : '');
      }) || []
    };
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

export async function getWorkflowSummary(queryId: string): Promise<Record<string, unknown>> {
  const response = await axios.get(`${API_CONFIG.BASE_URL}/workflow/${queryId}/summary`);
  return response.data;
}
