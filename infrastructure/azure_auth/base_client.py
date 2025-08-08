"""
Simple Base Azure Client - CODING_STANDARDS Compliant
Clean base class for Azure services without over-engineering.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from config.settings import azure_settings

logger = logging.getLogger(__name__)


class BaseAzureClient(ABC):
    """
    Simple base class for Azure service clients following CODING_STANDARDS.md:
    - Data-Driven Everything: Uses Azure settings for configuration
    - Universal Design: Works with any Azure service
    - Mathematical Foundation: Simple retry logic
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize simple Azure client"""
        self.config = config or {}
        self.endpoint = None
        self.key = None
        self.use_managed_identity = azure_settings.use_managed_identity
        self._client = None

        logger.info(
            f"Azure client initialized with managed identity: {self.use_managed_identity}"
        )

    @abstractmethod
    def _get_default_endpoint(self) -> str:
        """Get default endpoint for the service"""
        pass

    @abstractmethod
    def _initialize_client(self):
        """Initialize the specific Azure service client"""
        pass

    @abstractmethod
    def _health_check(self) -> bool:
        """Perform service health check"""
        pass

    def ensure_initialized(self):
        """Ensure client is initialized with simple setup"""
        if self._client is not None:
            return

        # Simple initialization
        self.endpoint = (
            self.config.get("endpoint")
            or self._get_default_endpoint()
        )

        if not self.use_managed_identity:
            self.key = self.config.get("api_key") or getattr(azure_settings, 'azure_openai_api_key', None)

        self._initialize_client()

        # Simple health check
        if not self._health_check():
            logger.warning(f"Health check failed for {self.__class__.__name__}")

    def create_success_response(self, operation: str, data: Any) -> Dict[str, Any]:
        """Create simple success response"""
        return {"success": True, "operation": operation, "data": data}

    def handle_azure_error(self, operation: str, error: Exception) -> Dict[str, Any]:
        """Simple error handling without over-engineering"""
        error_msg = f"{operation} failed: {str(error)}"
        logger.error(error_msg)

        return {"success": False, "operation": operation, "error": error_msg}

    def get_service_status(self) -> Dict[str, Any]:
        """Get simple service status"""
        try:
            self.ensure_initialized()
            is_healthy = self._health_check()

            return {
                "service": self.__class__.__name__,
                "status": "healthy" if is_healthy else "unhealthy",
                "endpoint": self.endpoint,
                "auth_type": (
                    "managed_identity" if self.use_managed_identity else "api_key"
                ),
            }
        except Exception as e:
            return {
                "service": self.__class__.__name__,
                "status": "error",
                "error": str(e),
            }
