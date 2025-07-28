"""
Base Azure Client - Common patterns for all Azure services
Consolidates initialization, error handling, and configuration
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from config.settings import azure_settings

logger = logging.getLogger(__name__)


class BaseAzureClient(ABC):
    """Base class for all Azure service clients"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with common Azure patterns"""
        self.config = config or {}
        self._client = None
        self._initialized = False
        
        # Standard Azure configuration loading
        self.endpoint = self.config.get('endpoint') or self._get_default_endpoint()
        self.key = self.config.get('key') or self._get_default_key()
        
        if not self.endpoint or not self.key:
            raise ValueError(f"{self.__class__.__name__} requires endpoint and key")
            
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def _get_default_endpoint(self) -> str:
        """Get default endpoint from settings"""
        pass
        
    @abstractmethod  
    def _get_default_key(self) -> str:
        """Get default key from settings"""
        pass
        
    @abstractmethod
    def _initialize_client(self):
        """Initialize the specific Azure service client"""
        pass
        
    def ensure_initialized(self):
        """Lazy initialization pattern"""
        if not self._initialized:
            self._initialize_client()
            self._initialized = True
            
    def handle_azure_error(self, operation: str, error: Exception) -> Dict[str, Any]:
        """Standard Azure error handling"""
        error_msg = f"{operation} failed: {str(error)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'operation': operation,
            'service': self.__class__.__name__
        }
        
    def create_success_response(self, operation: str, data: Any = None) -> Dict[str, Any]:
        """Standard success response"""
        return {
            'success': True,
            'operation': operation,
            'service': self.__class__.__name__,
            'data': data
        }