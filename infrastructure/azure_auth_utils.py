"""
Azure Authentication Utilities
=============================

Centralized Azure authentication to eliminate duplicate credential initialization
patterns found across 12+ files in the infrastructure layer.
"""

import logging
import os
from typing import Optional
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)

# Global credential instance for reuse
_azure_credential: Optional[DefaultAzureCredential] = None


def get_azure_credential() -> DefaultAzureCredential:
    """
    Get centralized Azure credential instance.
    
    Uses singleton pattern to avoid recreating DefaultAzureCredential
    multiple times across different Azure service clients.
    
    Returns:
        DefaultAzureCredential: Configured Azure credential
    """
    global _azure_credential
    
    if _azure_credential is None:
        try:
            # Create credential with managed identity preference
            _azure_credential = DefaultAzureCredential()
            logger.info("Azure credential initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure credential: {e}")
            raise
    
    return _azure_credential


def get_azure_credential_with_fallback() -> DefaultAzureCredential:
    """
    Get Azure credential with fallback options for development scenarios.
    
    Returns:
        DefaultAzureCredential: Configured Azure credential with fallback
    """
    try:
        return get_azure_credential()
    except Exception as e:
        logger.warning(f"Primary credential failed, attempting fallback: {e}")
        
        # For development environments, try CLI credential explicitly
        if os.getenv("AZURE_DEVELOPMENT_MODE", "false").lower() == "true":
            from azure.identity import AzureCliCredential
            try:
                return AzureCliCredential()
            except Exception:
                pass
        
        # Re-raise original exception if all fallbacks fail
        raise e


def reset_azure_credential():
    """
    Reset the global credential instance.
    
    Useful for testing or when credential needs to be refreshed.
    """
    global _azure_credential
    _azure_credential = None
    logger.info("Azure credential reset")