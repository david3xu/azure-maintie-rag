// Universal RAG domain types matching current backend API

export interface UniversalRAGRequest {
  query: string;
  max_results?: number;
  use_domain_analysis?: boolean;
}

export interface SearchResult {
  title: string;
  content: string;
  score: number;
  source: string;
  metadata?: Record<string, any>;
}

export interface UniversalRAGResponse {
  success: boolean;
  query: string;
  results: SearchResult[];
  total_results: number;
  processing_time: number;
  strategy_used: string;
  confidence_score: number;
  error?: string;
}

// Domain characteristics and processing
export interface DomainCharacteristics {
  domain_signature: string;
  vocabulary_complexity: number;
  concept_density: number;
  relationship_patterns: string[];
}

// Add more universal domain types here as needed for Universal RAG
