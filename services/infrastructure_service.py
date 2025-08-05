"""
Simple Infrastructure Service - CODING_STANDARDS Compliant
Clean infrastructure service without over-engineering enterprise patterns.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SimpleInfrastructureService:
    """
    Simple infrastructure service following CODING_STANDARDS.md:
    - Data-Driven Everything: Uses simple service monitoring
    - Universal Design: Works with any infrastructure component
    - Mathematical Foundation: Simple health calculations
    """

    def __init__(self):
        """Initialize simple infrastructure service"""
        self.services = {}  # service_name -> service_instance
        self.service_status = {}  # service_name -> status_info
        logger.info("Simple infrastructure service initialized")

    def register_service(self, service_name: str, service_instance: Any) -> Dict[str, Any]:
        """Register an infrastructure service"""
        try:
            self.services[service_name] = service_instance
            self.service_status[service_name] = {
                "status": "registered",
                "healthy": False,
                "last_check": None,
                "error": None
            }
            
            logger.info(f"Infrastructure service registered: {service_name}")
            return {
                "success": True,
                "service_name": service_name,
                "message": "Service registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {e}")
            return {
                "success": False,
                "service_name": service_name,
                "error": str(e)
            }

    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        try:
            if service_name not in self.services:
                return {
                    "success": False,
                    "service_name": service_name,
                    "error": "Service not registered"
                }

            service = self.services[service_name]
            
            # Try to call health check method
            healthy = False
            error = None
            
            try:
                if hasattr(service, 'health_check'):
                    result = await service.health_check()
                    healthy = isinstance(result, dict) and result.get("success", False)
                elif hasattr(service, '_health_check'):
                    healthy = service._health_check()
                elif hasattr(service, 'get_service_status'):
                    result = service.get_service_status()
                    healthy = isinstance(result, dict) and result.get("status") == "healthy"
                else:
                    # If no health check method, assume healthy if service exists
                    healthy = True
                    
            except Exception as health_error:
                healthy = False
                error = str(health_error)
            
            # Update service status
            from datetime import datetime
            self.service_status[service_name] = {
                "status": "healthy" if healthy else "unhealthy",
                "healthy": healthy,
                "last_check": datetime.now(),
                "error": error
            }
            
            return {
                "success": True,
                "service_name": service_name,
                "healthy": healthy,
                "status": "healthy" if healthy else "unhealthy",
                "error": error
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            return {
                "success": False,
                "service_name": service_name,
                "healthy": False,
                "error": str(e)
            }

    async def check_all_services_health(self) -> Dict[str, Any]:
        """Check health of all registered services"""
        try:
            health_results = {}
            healthy_count = 0
            total_count = len(self.services)
            
            for service_name in self.services.keys():
                result = await self.check_service_health(service_name)
                health_results[service_name] = result
                if result.get("healthy", False):
                    healthy_count += 1
            
            overall_healthy = healthy_count == total_count and total_count > 0
            health_percentage = (healthy_count / max(1, total_count)) * 100
            
            return {
                "success": True,
                "overall_healthy": overall_healthy,
                "healthy_services": healthy_count,
                "total_services": total_count,
                "health_percentage": round(health_percentage, 2),
                "service_results": health_results
            }
            
        except Exception as e:
            logger.error(f"Failed to check all services health: {e}")
            return {
                "success": False,
                "error": str(e),
                "overall_healthy": False
            }

    def get_service_info(self, service_name: str) -> Dict[str, Any]:
        """Get information about a specific service"""
        try:
            if service_name not in self.services:
                return {
                    "success": False,
                    "service_name": service_name,
                    "error": "Service not registered"
                }

            service = self.services[service_name]
            status = self.service_status.get(service_name, {})
            
            return {
                "success": True,
                "service_name": service_name,
                "service_type": type(service).__name__,
                "status": status.get("status", "unknown"),
                "healthy": status.get("healthy", False),
                "last_check": status.get("last_check").isoformat() if status.get("last_check") else None,
                "error": status.get("error")
            }
            
        except Exception as e:
            logger.error(f"Failed to get service info for {service_name}: {e}")
            return {
                "success": False,
                "service_name": service_name,
                "error": str(e)
            }

    def list_services(self) -> Dict[str, Any]:
        """List all registered services"""
        try:
            services_info = []
            
            for service_name in self.services.keys():
                service_info = self.get_service_info(service_name)
                if service_info.get("success", False):
                    services_info.append({
                        "service_name": service_name,
                        "service_type": service_info["service_type"],
                        "status": service_info["status"],
                        "healthy": service_info["healthy"]
                    })
            
            return {
                "success": True,
                "services": services_info,
                "total_count": len(services_info)
            }
            
        except Exception as e:
            logger.error(f"Failed to list services: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def remove_service(self, service_name: str) -> Dict[str, Any]:
        """Remove a service from the registry"""
        try:
            if service_name not in self.services:
                return {
                    "success": False,
                    "service_name": service_name,
                    "error": "Service not registered"
                }

            del self.services[service_name]
            if service_name in self.service_status:
                del self.service_status[service_name]
            
            logger.info(f"Infrastructure service removed: {service_name}")
            return {
                "success": True,
                "service_name": service_name,
                "message": "Service removed successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to remove service {service_name}: {e}")
            return {
                "success": False,
                "service_name": service_name,
                "error": str(e)
            }


# Backward compatibility - Global instance
_infrastructure_service = SimpleInfrastructureService()

# Backward compatibility functions
def register_infrastructure_service(service_name: str, service_instance: Any) -> Dict[str, Any]:
    """Backward compatibility function"""
    return _infrastructure_service.register_service(service_name, service_instance)

async def check_infrastructure_health(service_name: str = None) -> Dict[str, Any]:
    """Backward compatibility function"""
    if service_name:
        return await _infrastructure_service.check_service_health(service_name)
    else:
        return await _infrastructure_service.check_all_services_health()

def list_infrastructure_services() -> Dict[str, Any]:
    """Backward compatibility function"""
    return _infrastructure_service.list_services()

# Backward compatibility aliases
InfrastructureService = SimpleInfrastructureService
ConsolidatedInfrastructureService = SimpleInfrastructureService