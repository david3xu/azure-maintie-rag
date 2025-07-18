import axios from 'axios';
import type { QueryRequest, QueryResponse, StreamingQueryResponse } from '../types/api';

const API_BASE = 'http://localhost:8000/api/v1';

export async function postStructuredQuery(query: string): Promise<QueryResponse> {
  const response = await axios.post<QueryResponse>(
    `${API_BASE}/query/structured/`,
    {
      query: query.trim(),
      max_results: 10,
      include_explanations: true,
      enable_safety_warnings: true,
    } as QueryRequest
  );
  return response.data;
}

export async function postStreamingQuery(query: string): Promise<StreamingQueryResponse> {
  const response = await axios.post<StreamingQueryResponse>(
    `${API_BASE}/query/streaming`,
    {
      query: query.trim(),
      max_results: 10,
      include_explanations: true,
      enable_safety_warnings: true,
    } as QueryRequest
  );
  return response.data;
}

export async function getWorkflowSummary(queryId: string): Promise<any> {
  const response = await axios.get(`${API_BASE}/workflow/${queryId}/summary`);
  return response.data;
}
