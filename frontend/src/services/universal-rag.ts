import axios from 'axios';
import type { UniversalRAGRequest, UniversalRAGResponse } from '../types/domain';
import { API_CONFIG } from '../utils/api-config';

export async function fetchUniversalRAG(request: UniversalRAGRequest): Promise<UniversalRAGResponse> {
  // Map UniversalRAGRequest to our SearchRequest format
  const searchRequest = {
    query: request.query,
    max_results: request.max_results || 10,
    use_domain_analysis: true,
    include_agent_metrics: false
  };

  const response = await axios.post(
    `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.SEARCH}`, 
    searchRequest,
    {
      timeout: 60000,
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );
  
  // Map the backend SearchResponse to UniversalRAGResponse format
  const searchData = response.data;
  return {
    success: searchData.success,
    query: searchData.query,
    results: searchData.results || [],
    total_results: searchData.total_results_found || 0,
    processing_time: searchData.execution_time,
    strategy_used: searchData.strategy_used || 'universal-search',
    confidence_score: searchData.search_confidence || 0,
    error: searchData.error
  };
}
