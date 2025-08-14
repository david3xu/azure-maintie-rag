import axios from "axios";
import type {
  QueryResponse,
  KnowledgeExtractionRequest,
  KnowledgeExtractionResponse,
  HealthResponse,
} from "../types/api";
import { API_CONFIG } from "../utils/api-config";

export async function postUniversalQuery(
  query: string
  // Domain parameter removed - violates zero hardcoded domain bias rule
): Promise<QueryResponse> {
  try {
    console.log(
      "Making API request to unified RAG endpoint:",
      `${API_CONFIG.BASE_URL}/api/v1/rag`
    );

    const ragRequest = {
      query: query.trim(),
      max_results: 10,
      max_tokens: 1000,
      use_domain_analysis: true,
      include_sources: true,
      include_search_results: true,
    };

    console.log("Request payload:", ragRequest);

    const response = await axios.post(
      `${API_CONFIG.BASE_URL}/api/v1/rag`,
      ragRequest,
      {
        timeout: 90000, // Increased timeout for search + answer generation
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    console.log("API response received:", response.data);
    const backendData = response.data;

    if (!backendData.success) {
      throw new Error(backendData.error || "RAG request failed");
    }

    // Now we have a real generated answer instead of just search results summary
    return {
      query: backendData.query,
      generated_response: backendData.generated_answer, // Real Azure OpenAI generated answer
      confidence_score: backendData.confidence_score,
      processing_time: backendData.execution_time,
      safety_warnings: [], // Can be added in future
      sources: backendData.sources_used,
      citations: backendData.search_results?.map((r: any) => {
        const content = r.content || "";
        return content.substring(0, 200) + (content.length > 200 ? "..." : "");
      }) || [],
    };
  } catch (error) {
    console.error("Universal RAG query failed:", error);
    if (axios.isAxiosError(error)) {
      const message = error.response?.data?.error || error.message;
      throw new Error(`RAG processing failed: ${message}`);
    }
    throw error;
  }
}

export async function postKnowledgeExtraction(
  content: string
): Promise<KnowledgeExtractionResponse> {
  try {
    const extractRequest: KnowledgeExtractionRequest = {
      content: content,
      use_domain_analysis: true,
    };

    const response = await axios.post<KnowledgeExtractionResponse>(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.EXTRACT}`,
      extractRequest,
      {
        timeout: 60000,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    return response.data;
  } catch (error) {
    console.error("Knowledge extraction request failed:", error);
    throw error;
  }
}

export async function getHealthStatus(): Promise<HealthResponse> {
  try {
    const response = await axios.get<HealthResponse>(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.HEALTH}`,
      { timeout: 30000 }
    );
    return response.data;
  } catch (error) {
    console.error("Health check request failed:", error);
    throw error;
  }
}

export async function getApiInfo(): Promise<Record<string, any>> {
  try {
    const response = await axios.get(
      `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.ROOT}`,
      { timeout: 10000 }
    );
    return response.data;
  } catch (error) {
    console.error("API info request failed:", error);
    throw error;
  }
}

//  NO FAKE FALLBACK - QUICK FAIL if endpoint doesn't exist
export async function getWorkflowSummary(
  queryId: string
): Promise<Record<string, unknown>> {
  try {
    const response = await axios.get(
      `${API_CONFIG.BASE_URL}/api/v1/workflow/summary/${queryId}`,
      { timeout: 30000 }
    );
    return response.data;
  } catch (error) {
    console.error(
      "REAL workflow summary request FAILED - no fake fallback:",
      error
    );
    throw new Error(
      "Real Azure workflow summary endpoint not available - QUICK FAIL mode"
    );
  }
}
