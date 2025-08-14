// Runtime API configuration - detects REAL backend URL
function getApiBaseUrl(): string {
  // 1. Try build-time environment variable first
  const buildTimeUrl = import.meta.env.VITE_API_URL;
  if (buildTimeUrl && buildTimeUrl !== 'http://localhost:8000') {
    return buildTimeUrl;
  }
  
  // 2. For Container Apps, detect based on current hostname pattern
  const hostname = window.location.hostname;
  if (hostname.includes('.azurecontainerapps.io')) {
    // Replace 'frontend' with 'backend' in the Container App URL
    const backendUrl = hostname.replace('ca-frontend-', 'ca-backend-');
    return `https://${backendUrl}`;
  }
  
  // 3. Development fallback
  return 'http://localhost:8000';
}

export const API_CONFIG = {
  BASE_URL: getApiBaseUrl(),
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