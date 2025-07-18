// Chat message type for Universal RAG chat history
import type { QueryResponse } from './api';

export interface ChatMessage {
  id: string;
  query: string;
  response: QueryResponse | null;
  timestamp: Date;
  isStreaming: boolean;
  queryId: string | null;
  error: string | null;
}
