import { useCallback, useState } from 'react';
import { fetchUniversalRAG } from '../services/universal-rag';
import type { UniversalRAGRequest, UniversalRAGResponse } from '../types/domain';

// Hook for orchestrating Universal RAG API calls and state
export function useUniversalRAG() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<UniversalRAGResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runUniversalRAG = useCallback(async (request: UniversalRAGRequest) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchUniversalRAG(request);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  return { loading, result, error, runUniversalRAG };
}
