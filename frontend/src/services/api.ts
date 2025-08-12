import axios from 'axios';
import type { QueryResponse } from '../types/api';
import { API_CONFIG } from '../utils/api-config';

// Request types matching our FastAPI backend
interface SearchRequest {
  query: string;
  max_results?: number;
  use_domain_analysis?: boolean;
  include_agent_metrics?: boolean;
}

interface KnowledgeExtractionRequest {
  content: string;
  use_domain_analysis?: boolean;
}

// Response types matching our FastAPI backend
interface SearchResult {
  title: string;
  content: string;
  score: number;
  source: string;
  metadata?: Record<string, any>;
}

interface SearchResponse {
  success: boolean;
  query: string;
  results: SearchResult[];
  total_results_found: number;
  search_confidence: number;
  strategy_used: string;
  execution_time: number;
  timestamp: string;
  agent_metrics?: Record<string, any>;
  error?: string;
}

interface KnowledgeExtractionResponse {
  success: boolean;
  entities: Array<Record<string, any>>;
  relationships: Array<Record<string, any>>;
  extraction_confidence: number;
  processing_signature: string;
  execution_time: number;
  timestamp: string;
  error?: string;
}

interface HealthResponse {
  status: string;
  services_available: string[];
  total_services: number;
  agent_status: Record<string, string>;
  timestamp: string;
}

export async function postUniversalQuery(
  query: string,
  _domain: string = 'general' // Prefix with underscore to indicate intentionally unused
): Promise<QueryResponse> {
  try {
    console.log('Making API request to:', `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.SEARCH}`);
    
    const searchRequest: SearchRequest = {
      query: query.trim(),
      max_results: 10,
      use_domain_analysis: true,
      include_agent_metrics: true
    };
    
    console.log('Request payload:', searchRequest);

    const response = await axios.post<SearchResponse>(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.SEARCH}`,
      searchRequest,
      {
        timeout: 60000, // 60 second timeout
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    console.log('API response received:', response.data);
    const backendData = response.data;

    // Convert backend SearchResponse to frontend QueryResponse format
    return {
      query: backendData.query,
      generated_response: backendData.success 
        ? `Found ${backendData.total_results_found} results using ${backendData.strategy_used} strategy`
        : backendData.error || 'Search failed',
      confidence_score: backendData.search_confidence,
      processing_time: backendData.execution_time,
      safety_warnings: [], // Our backend doesn't return safety warnings yet
      sources: backendData.results?.map(r => r.source) || [],
      citations: backendData.results?.map(r => {
        const content = r.content || '';
        return content.substring(0, 100) + (content.length > 100 ? '...' : '');
      }) || []
    };
    
  } catch (error) {
    console.error('API request failed:', error);
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