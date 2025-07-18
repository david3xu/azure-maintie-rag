import axios from 'axios';
import type { UniversalRAGRequest, UniversalRAGResponse } from '../types/domain';
import { API_CONFIG } from '../utils/api-config';

export async function fetchUniversalRAG(request: UniversalRAGRequest): Promise<UniversalRAGResponse> {
  const response = await axios.post(`${API_CONFIG.BASE_URL}/universal-rag`, request);
  return response.data as UniversalRAGResponse;
}
