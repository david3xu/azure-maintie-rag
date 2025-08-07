"""
Simplified Universal Search Agent - PydanticAI Best Practices
============================================================

This implementation demonstrates simplified search orchestration:
- Direct search operations without complex orchestrators
- Simple result synthesis
- Clean agent-to-agent communication
- Focus on core search functionality
"""

import os
import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Simple, focused dependencies
class SearchDeps(BaseModel):
    """Simple dependencies for universal search"""
    max_results: int = 10
    similarity_threshold: float = 0.7
    enable_vector_search: bool = True
    enable_graph_search: bool = True
    enable_gnn_search: bool = False  # Simplified - can be enabled later

# Clean output models
class SearchResult(BaseModel):
    """Individual search result"""
    content: str
    relevance_score: float
    source: str
    result_type: str  # "vector", "graph", "gnn"
    metadata: Dict[str, any] = {}

class UniversalSearchResult(BaseModel):
    """Universal search results with synthesis"""
    query: str
    results: List[SearchResult]
    synthesis_score: float
    execution_time: float
    modalities_used: List[str]
    total_results: int

# Lazy agent initialization
_agent = None

def _create_search_agent():
    """Internal agent creation with tools"""
    # Direct model configuration from environment
    model_name = f"openai:{os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o')}"
    
    agent = Agent(
        model_name,
        deps_type=SearchDeps,
        result_type=UniversalSearchResult,
        system_prompt="""You are a Universal Search Agent that orchestrates multi-modal search operations.
    
        Your capabilities:
        1. Execute vector similarity search for semantic matching
        2. Perform graph traversal search for relationship discovery  
        3. Synthesize results from multiple search modalities
        4. Rank and filter results based on relevance
        
        Always return structured UniversalSearchResult with synthesized findings.""",
    )

@agent.tool
async def vector_search(ctx: RunContext[SearchDeps], query: str) -> List[Dict[str, any]]:
    """Perform vector similarity search"""
    
    if not ctx.deps.enable_vector_search:
        return []
    
    # Simplified vector search simulation
    # In production, this would interface with Azure Cognitive Search
    results = []
    
    # Mock vector search results based on query analysis
    query_tokens = query.lower().split()
    
    # Simulate finding relevant documents (would be actual vector search)
    mock_documents = [
        {"content": "Programming concepts and methodologies", "score": 0.9},
        {"content": "Software development best practices", "score": 0.85},
        {"content": "Computer science fundamentals", "score": 0.8},
        {"content": "Algorithm design patterns", "score": 0.75},
        {"content": "System architecture principles", "score": 0.7}
    ]
    
    for doc in mock_documents:
        if doc["score"] >= ctx.deps.similarity_threshold:
            results.append({
                "content": doc["content"],
                "relevance_score": doc["score"],
                "source": "vector_index",
                "result_type": "vector"
            })
            
            if len(results) >= ctx.deps.max_results:
                break
    
    return results

@agent.tool
async def graph_search(ctx: RunContext[SearchDeps], query: str) -> List[Dict[str, any]]:
    """Perform graph traversal search"""
    
    if not ctx.deps.enable_graph_search:
        return []
    
    # Simplified graph search simulation  
    # In production, this would interface with Azure Cosmos DB Gremlin
    results = []
    
    # Mock graph traversal results
    mock_graph_results = [
        {"content": "Related concept through entity relationships", "score": 0.8},
        {"content": "Connected knowledge via graph paths", "score": 0.75},
        {"content": "Relationship-based knowledge discovery", "score": 0.7}
    ]
    
    for result in mock_graph_results:
        if result["score"] >= ctx.deps.similarity_threshold:
            results.append({
                "content": result["content"],
                "relevance_score": result["score"],
                "source": "knowledge_graph",
                "result_type": "graph"
            })
    
    return results[:ctx.deps.max_results]

@agent.tool
async def synthesize_results(
    ctx: RunContext[SearchDeps], 
    vector_results: List[Dict[str, any]], 
    graph_results: List[Dict[str, any]]
) -> Dict[str, any]:
    """Synthesize results from multiple search modalities"""
    
    all_results = vector_results + graph_results
    
    # Simple result deduplication and ranking
    seen_content = set()
    unique_results = []
    
    for result in sorted(all_results, key=lambda x: x["relevance_score"], reverse=True):
        content_key = result["content"][:50]  # Simple deduplication
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_results.append(result)
    
    # Calculate synthesis score based on result quality and diversity
    if unique_results:
        avg_score = sum(r["relevance_score"] for r in unique_results) / len(unique_results)
        modality_diversity = len(set(r["result_type"] for r in unique_results))
        synthesis_score = avg_score * (1 + 0.1 * modality_diversity)  # Bonus for diversity
    else:
        synthesis_score = 0.0
    
    return {
        "synthesized_results": unique_results[:ctx.deps.max_results],
        "synthesis_score": min(synthesis_score, 1.0),
        "result_count": len(unique_results)
    }

    return agent

def create_search_agent() -> Agent[SearchDeps, UniversalSearchResult]:
    """Create universal search agent with PydanticAI best practices"""
    global _agent
    if _agent is None:
        _agent = _create_search_agent()
    return _agent

# Simple factory function
def get_search_agent() -> Agent[SearchDeps, UniversalSearchResult]:
    """Get universal search agent (no global state)"""
    return create_search_agent()

# Simple search function for backward compatibility  
async def universal_search(
    query: str,
    max_results: int = 10,
    similarity_threshold: float = 0.7
) -> UniversalSearchResult:
    """Simple universal search function"""
    
    start_time = time.time()
    agent = get_search_agent()
    deps = SearchDeps(
        max_results=max_results,
        similarity_threshold=similarity_threshold,
        enable_vector_search=True,
        enable_graph_search=True,
        enable_gnn_search=False
    )
    
    # Run search with the agent
    result = await agent.run(
        f"Search for: {query}",
        deps=deps
    )
    
    processing_time = time.time() - start_time
    
    # The agent will return UniversalSearchResult directly
    return result.data

# Export simplified interface
__all__ = [
    "create_search_agent",
    "get_search_agent", 
    "universal_search",
    "SearchDeps",
    "UniversalSearchResult", 
    "SearchResult"
]