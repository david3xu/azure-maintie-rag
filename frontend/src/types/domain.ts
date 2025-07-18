// Universal RAG domain types

export interface UniversalRAGRequest {
  query: string;
  domain: string;
  // Add more fields as needed for Universal RAG
}

export interface UniversalRAGResponse {
  answer: string;
  sources?: Array<{ title: string; url: string }>;
  // Add more fields as needed for Universal RAG
}

// Add more universal domain types here as needed for Universal RAG
