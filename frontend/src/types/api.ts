// API request/response types for Universal RAG
export interface QueryRequest {
  query: string;
  domain?: string;
  max_results?: number;
  include_explanations?: boolean;
  enable_safety_warnings?: boolean;
}

export interface QueryResponse {
  query: string;
  generated_response: string;
  confidence_score: number;
  processing_time: number;
  safety_warnings: string[];
  sources: string[];
  citations: string[];
}

export interface StreamingQueryResponse {
  query_id: string;
  status: string;
  stream_url?: string;
}

export interface UniversalQueryRequest {
  query: string;
  domain: string;
  max_results: number;
  include_explanations: boolean;
  enable_safety_warnings: boolean;
}

export interface UniversalQueryResponse {
  success: boolean;
  query: string;
  domain: string;
  generated_response: {
    answer: string;
    explanation?: string;
    confidence?: number;
  };
  search_results: Array<{
    content: string;
    source: string;
    score: number;
  }>;
  processing_time: number;
  system_stats: Record<string, any>;
  timestamp: string;
  error?: string;
}
