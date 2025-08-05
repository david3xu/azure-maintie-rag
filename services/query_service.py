"""
Simple Query Service - CODING_STANDARDS Compliant
Clean query service without over-engineering abstractions.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SimpleQueryService:
    """
    Simple query service following CODING_STANDARDS.md:
    - Data-Driven Everything: Uses direct agent calls without abstractions
    - Universal Design: Works with any query type
    - Mathematical Foundation: Simple query processing patterns
    """

    def __init__(self):
        """Initialize simple query service"""
        self.agents = {}
        self.query_cache = {}
        logger.info("Simple query service initialized")

    def register_agent(self, agent_name: str, agent_instance: Any) -> None:
        """Register an agent for query processing"""
        self.agents[agent_name] = agent_instance
        logger.info(f"Query agent registered: {agent_name}")

    async def process_query(self, query: str, agent_name: str = "default", domain: str = "general") -> Dict[str, Any]:
        """Process query using simple approach"""
        try:
            # Check cache first
            cache_key = f"{agent_name}:{domain}:{hash(query)}"
            if cache_key in self.query_cache:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return self.query_cache[cache_key]

            # Get agent
            if agent_name not in self.agents:
                return {
                    "success": False,
                    "error": f"Agent not found: {agent_name}",
                    "query": query,
                    "agent_name": agent_name
                }

            agent = self.agents[agent_name]

            # Process query
            request = {
                "query": query,
                "domain": domain,
                "agent_name": agent_name
            }

            if hasattr(agent, 'process_query'):
                result = await agent.process_query(request)
            elif hasattr(agent, 'process_request'):
                result = await agent.process_request(request)
            else:
                return {
                    "success": False,
                    "error": f"Agent {agent_name} has no query processing method",
                    "query": query
                }

            # Cache successful results
            response = {
                "success": True,
                "query": query,
                "agent_name": agent_name,
                "domain": domain,
                "result": result
            }
            
            self.query_cache[cache_key] = response
            
            return response

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "agent_name": agent_name
            }

    async def search_documents(self, query: str, domain: str = "general", limit: int = 10) -> Dict[str, Any]:
        """Search documents using simple approach"""
        try:
            # Use search agent if available
            if "search" in self.agents:
                search_request = {
                    "query": query,
                    "domain": domain,
                    "limit": limit,
                    "search_type": "documents"
                }
                return await self.process_query(query, "search", domain)
            else:
                return {
                    "success": False,
                    "error": "Search agent not available",
                    "query": query
                }

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    async def extract_knowledge(self, text: str, domain: str = "general") -> Dict[str, Any]:
        """Extract knowledge using simple approach"""
        try:
            # Use knowledge extraction agent if available
            if "knowledge_extraction" in self.agents:
                extraction_request = {
                    "text": text,
                    "domain": domain,
                    "operation": "extract_knowledge"
                }
                return await self.process_query(text, "knowledge_extraction", domain)
            else:
                return {
                    "success": False,
                    "error": "Knowledge extraction agent not available",
                    "text": text[:100] + "..." if len(text) > 100 else text
                }

        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": text[:100] + "..." if len(text) > 100 else text
            }

    def clear_cache(self) -> Dict[str, Any]:
        """Clear query cache"""
        cache_size = len(self.query_cache)
        self.query_cache.clear()
        logger.info(f"Query cache cleared: {cache_size} entries removed")
        return {
            "success": True,
            "message": f"Cache cleared: {cache_size} entries removed"
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.query_cache),
            "cache_keys": list(self.query_cache.keys())[:10]  # Show first 10 keys
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "status": "healthy",
            "registered_agents": list(self.agents.keys()),
            "cache_size": len(self.query_cache),
            "total_agents": len(self.agents)
        }


# Backward compatibility - Global instance
_query_service = SimpleQueryService()

# Backward compatibility functions
async def process_query(query: str, agent_name: str = "default", domain: str = "general") -> Dict[str, Any]:
    """Backward compatibility function"""
    return await _query_service.process_query(query, agent_name, domain)

async def search_documents(query: str, domain: str = "general", limit: int = 10) -> Dict[str, Any]:
    """Backward compatibility function"""
    return await _query_service.search_documents(query, domain, limit)

async def extract_knowledge(text: str, domain: str = "general") -> Dict[str, Any]:
    """Backward compatibility function"""
    return await _query_service.extract_knowledge(text, domain)

def register_query_agent(agent_name: str, agent_instance: Any) -> None:
    """Backward compatibility function"""
    _query_service.register_agent(agent_name, agent_instance)

# Backward compatibility aliases
QueryService = SimpleQueryService
ConsolidatedQueryService = SimpleQueryService
EnhancedQueryService = SimpleQueryService