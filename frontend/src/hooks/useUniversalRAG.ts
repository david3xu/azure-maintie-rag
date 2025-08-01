import { useState } from 'react';
import { postUniversalQuery } from '../services/api';

export const useUniversalRAG = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const runUniversalRAG = async (request: { query: string; domain: string }) => {
    setLoading(true);
    setError(null);
    try {
      const response = await postUniversalQuery(request.query, request.domain);
      setResult(response);
      return response;
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    loading,
    result,
    error,
    runUniversalRAG
  };
};
