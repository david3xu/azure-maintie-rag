export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  ENDPOINTS: {
    SEARCH: '/api/v1/search',
    EXTRACT: '/api/v1/extract', 
    HEALTH: '/api/v1/health',
    ROOT: '/',
    // REAL Azure streaming endpoint - matches backend implementation
    WORKFLOW_STREAM: '/api/v1/stream/workflow'
  }
};

console.log('API Configuration:', {
  baseUrl: API_CONFIG.BASE_URL,
  searchEndpoint: `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.SEARCH}`,
  streamEndpoint: `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.WORKFLOW_STREAM}`
});