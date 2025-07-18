// API request/response types for Universal RAG
export interface QueryRequest {
  query: string;
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
  stream_url: string;
}
