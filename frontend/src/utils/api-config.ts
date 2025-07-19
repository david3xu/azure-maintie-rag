export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  ENDPOINTS: {
    UNIVERSAL_QUERY: '/api/v1/query/universal',
    STREAMING_QUERY: '/api/v1/query/streaming',
    STREAM: '/api/v1/query/stream',
    REAL_WORKFLOW_STREAM: '/api/v1/query/stream/real'
  }
};

console.log('API URL:', `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.UNIVERSAL_QUERY}`);