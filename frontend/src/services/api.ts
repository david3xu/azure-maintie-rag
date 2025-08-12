import axios from 'axios';
import type { 
  QueryResponse, 
  SearchRequest, 
  SearchResponse, 
  KnowledgeExtractionRequest,
  KnowledgeExtractionResponse,
  HealthResponse 
} from '../types/api';
import { API_CONFIG } from '../utils/api-config';

export async function postUniversalQuery(
  query: string,
  _domain: string = 'general' // Domain parameter kept for compatibility but not used by backend
): Promise<QueryResponse> {
  try {
    console.log('Making API request to:', `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.SEARCH}`);
    
    const searchRequest: SearchRequest = {
      query: query.trim(),
      max_results: 10,
      use_domain_analysis: true,
      include_agent_metrics: false // Set to false for better performance
    };
    
    console.log('Request payload:', searchRequest);

    const response = await axios.post<SearchResponse>(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.SEARCH}`,
      searchRequest,
      {
        timeout: 60000, // 60 second timeout for Azure operations
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    console.log('API response received:', response.data);
    const backendData = response.data;

    if (!backendData.success) {
      throw new Error(backendData.error || 'Search request failed');
    }

    // Convert backend SearchResponse to frontend QueryResponse format
    const topResults = backendData.results.slice(0, 5); // Show top 5 results
    const resultSummary = topResults.length > 0 
      ? `Found ${backendData.total_results_found} results using ${backendData.strategy_used} strategy. Top results include: ${topResults.map(r => r.title).join(', ')}`
      : `Search completed using ${backendData.strategy_used} strategy but no results found.`;

    return {
      query: backendData.query,
      generated_response: resultSummary,
      confidence_score: backendData.search_confidence,
      processing_time: backendData.execution_time,
      safety_warnings: [], // Not implemented in backend yet
      sources: backendData.results.map(r => r.source),
      citations: backendData.results.map(r => {
        const content = r.content || '';
        return content.substring(0, 150) + (content.length > 150 ? '...' : '');
      })
    };
    
  } catch (error) {
    console.error('Universal query failed:', error);
    if (axios.isAxiosError(error)) {
      const message = error.response?.data?.error || error.message;
      throw new Error(`Search failed: ${message}`);
    }
    throw error;
  }
}

export async function postKnowledgeExtraction(content: string): Promise<KnowledgeExtractionResponse> {
  try {
    const extractRequest: KnowledgeExtractionRequest = {
      content: content,
      use_domain_analysis: true
    };

    const response = await axios.post<KnowledgeExtractionResponse>(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.EXTRACT}`,
      extractRequest,
      {
        timeout: 60000,
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    return response.data;
  } catch (error) {
    console.error('Knowledge extraction request failed:', error);
    throw error;
  }
}

export async function getHealthStatus(): Promise<HealthResponse> {
  try {
    const response = await axios.get<HealthResponse>(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.HEALTH}`,
      { timeout: 30000 }
    );
    return response.data;
  } catch (error) {
    console.error('Health check request failed:', error);
    throw error;
  }
}

export async function getApiInfo(): Promise<Record<string, any>> {
  try {
    const response = await axios.get(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.ROOT}`,
      { timeout: 10000 }
    );
    return response.data;
  } catch (error) {
    console.error('API info request failed:', error);
    throw error;
  }
}

// FUNC! NO FAKE FALLBACK - QUICK FAIL if endpoint doesn't exist
export async function getWorkflowSummary(queryId: string): Promise<Record<string, unknown>> {
  try {
    const response = await axios.get(
      `${API_CONFIG.BASE_URL}/api/v1/workflow/summary/${queryId}`,
      { timeout: 30000 }
    );
    return response.data;
  } catch (error) {
    console.error('REAL workflow summary request FAILED - no fake fallback:', error);
    throw new Error('Real Azure workflow summary endpoint not available - QUICK FAIL mode');
  }
}