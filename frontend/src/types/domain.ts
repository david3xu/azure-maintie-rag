// Universal RAG domain types

export interface UniversalRAGRequest {
  query: string;
  domain: string;
  max_results?: number;
  include_explanations?: boolean;
  enable_safety_warnings?: boolean;
}

export interface UniversalRAGResponse {
  success: boolean;
  query: string;
  domain: string;
  answer: string;
  sources?: Array<{ title: string; content: string }>;
  processing_time: number;
}

// Add more universal domain types here as needed for Universal RAG
