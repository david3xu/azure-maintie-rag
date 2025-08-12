// API request/response types matching backend FastAPI models

// Search Request - matches backend SearchRequest
export interface SearchRequest {
  query: string;
  max_results?: number;
  use_domain_analysis?: boolean;
  include_agent_metrics?: boolean;
}

// Search Result - matches backend SearchResult
export interface SearchResult {
  title: string;
  content: string;
  score: number;
  source: string;
  metadata?: Record<string, any>;
}

// Search Response - matches backend SearchResponse
export interface SearchResponse {
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

// Knowledge Extraction Request - matches backend
export interface KnowledgeExtractionRequest {
  content: string;
  use_domain_analysis?: boolean;
}

// Knowledge Extraction Response - matches backend
export interface KnowledgeExtractionResponse {
  success: boolean;
  entities: Array<Record<string, any>>;
  relationships: Array<Record<string, any>>;
  extraction_confidence: number;
  processing_signature: string;
  execution_time: number;
  timestamp: string;
  error?: string;
}

// Health Response - matches backend HealthResponse
export interface HealthResponse {
  status: string;
  services_available: string[];
  total_services: number;
  agent_status: Record<string, string>;
  timestamp: string;
}

// Legacy compatibility - keep for existing components
export interface QueryResponse {
  query: string;
  generated_response: string;
  confidence_score: number;
  processing_time: number;
  safety_warnings: string[];
  sources: string[];
  citations: string[];
}

// Workflow streaming events
export interface WorkflowEvent {
  event_type: 'connection_established' | 'progress' | 'workflow_completed' | 'workflow_failed' | 'error';
  query_id: string;
  timestamp?: string;
  step_number?: number;
  step_name?: string;
  user_friendly_name?: string;
  status?: 'in_progress' | 'completed' | 'failed';
  technology?: string;
  details?: string;
  progress_percentage?: number;
  processing_time_ms?: number;
  error?: string;
  // Completion event fields
  query?: string;
  generated_response?: string;
  confidence_score?: number;
  processing_time?: number;
  safety_warnings?: string[];
  sources?: string[];
  citations?: string[];
}
