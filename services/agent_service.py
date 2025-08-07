"""
Simple Agent Service - CODING_STANDARDS Compliant
Clean agent service without over-engineering abstractions.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SimpleAgentService:
    """
    Simple agent service following CODING_STANDARDS.md:
    - Data-Driven Everything: Uses agents directly without abstractions
    - Universal Design: Works with any agent type
    - Mathematical Foundation: Simple request/response patterns
    """

    def __init__(self):
        """Initialize simple agent service"""
        self.agents = {}
        logger.info("Simple agent service initialized")

    def register_agent(self, agent_name: str, agent_instance: Any) -> None:
        """Register an agent instance"""
        self.agents[agent_name] = agent_instance
        logger.info(f"Agent registered: {agent_name}")

    async def process_request(
        self, agent_name: str, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process request using specified agent"""
        try:
            if agent_name not in self.agents:
                return {
                    "success": False,
                    "error": f"Agent not found: {agent_name}",
                    "agent_name": agent_name,
                }

            agent = self.agents[agent_name]

            # Simple request processing
            if hasattr(agent, "process_query"):
                result = await agent.process_query(request)
            elif hasattr(agent, "process_request"):
                result = await agent.process_request(request)
            else:
                return {
                    "success": False,
                    "error": f"Agent {agent_name} has no process method",
                    "agent_name": agent_name,
                }

            return {"success": True, "agent_name": agent_name, "result": result}

        except Exception as e:
            logger.error(f"Agent request failed for {agent_name}: {e}")
            return {"success": False, "error": str(e), "agent_name": agent_name}

    async def get_agent_health(self, agent_name: str = None) -> Dict[str, Any]:
        """Get health status of agents"""
        try:
            if agent_name:
                # Single agent health
                if agent_name not in self.agents:
                    return {
                        "agent_name": agent_name,
                        "status": "not_found",
                        "healthy": False,
                    }

                agent = self.agents[agent_name]
                healthy = hasattr(agent, "process_query") or hasattr(
                    agent, "process_request"
                )

                return {
                    "agent_name": agent_name,
                    "status": "healthy" if healthy else "unhealthy",
                    "healthy": healthy,
                }
            else:
                # All agents health
                health_status = {}
                for name, agent in self.agents.items():
                    healthy = hasattr(agent, "process_query") or hasattr(
                        agent, "process_request"
                    )
                    health_status[name] = {
                        "status": "healthy" if healthy else "unhealthy",
                        "healthy": healthy,
                    }

                return {
                    "overall_status": (
                        "healthy"
                        if all(h["healthy"] for h in health_status.values())
                        else "partial"
                    ),
                    "agents": health_status,
                    "total_agents": len(self.agents),
                }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "total_agents": len(self.agents),
            }

    def list_agents(self) -> Dict[str, Any]:
        """List all registered agents"""
        return {"agents": list(self.agents.keys()), "count": len(self.agents)}

    def remove_agent(self, agent_name: str) -> Dict[str, Any]:
        """Remove an agent"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"Agent removed: {agent_name}")
            return {
                "success": True,
                "agent_name": agent_name,
                "message": "Agent removed",
            }
        else:
            return {
                "success": False,
                "error": f"Agent not found: {agent_name}",
                "agent_name": agent_name,
            }


# Backward compatibility - Global instance
_agent_service = SimpleAgentService()


# Backward compatibility functions
async def process_agent_request(
    agent_name: str, request: Dict[str, Any]
) -> Dict[str, Any]:
    """Backward compatibility function"""
    return await _agent_service.process_request(agent_name, request)


def register_agent(agent_name: str, agent_instance: Any) -> None:
    """Backward compatibility function"""
    _agent_service.register_agent(agent_name, agent_instance)


async def get_agent_health(agent_name: str = None) -> Dict[str, Any]:
    """Backward compatibility function"""
    return await _agent_service.get_agent_health(agent_name)


# Backward compatibility aliases
AgentService = SimpleAgentService
ConsolidatedAgentService = SimpleAgentService
