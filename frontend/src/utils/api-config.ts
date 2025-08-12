export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  ENDPOINTS: {
    SEARCH: '/api/v1/search',
    EXTRACT: '/api/v1/extract', 
    HEALTH: '/api/v1/health',
    ROOT: '/',
    // FUNC! REAL Azure streaming endpoint - NO fake endpoints
    REAL_WORKFLOW_STREAM: '/api/v1/stream/workflow'
  }
};

console.log('API URL:', `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.SEARCH}`);