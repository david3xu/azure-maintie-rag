"""
Azure Authentication Utilities with Session Management
======================================================

Centralized Azure authentication to eliminate duplicate credential initialization
patterns found across 12+ files in the infrastructure layer.
Enhanced with auto-refresh session management for production reliability.
"""

import logging
import os
import time
from typing import Optional

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

logger = logging.getLogger(__name__)

# Global credential instance and session tracking for reuse
_azure_credential: Optional[DefaultAzureCredential] = None
_last_refresh_time: float = 0
_refresh_threshold_minutes: int = 50  # Refresh before 1-hour Azure token expiry


def get_azure_credential() -> DefaultAzureCredential:
    """
    Get centralized Azure credential instance with auto-refresh session management.

    Uses singleton pattern to avoid recreating DefaultAzureCredential
    multiple times across different Azure service clients. Automatically
    refreshes credentials before expiry for production reliability.

    Returns:
        DefaultAzureCredential: Configured Azure credential with session management
    """
    global _azure_credential, _last_refresh_time

    current_time = time.time()
    time_since_refresh = (current_time - _last_refresh_time) / 60  # Convert to minutes

    # Refresh if credential doesn't exist or is approaching expiry
    if not _azure_credential or time_since_refresh > _refresh_threshold_minutes:
        _refresh_credential()

    return _azure_credential


def _refresh_credential():
    """
    Refresh Azure credential with DefaultAzureCredential (includes proper timeout handling).

    DefaultAzureCredential handles the authentication chain properly with timeouts,
    avoiding the 2-minute ManagedIdentityCredential timeout in non-managed environments.
    """
    global _azure_credential, _last_refresh_time

    try:
        # Use DefaultAzureCredential which includes proper timeout handling
        # and authentication chain (Environment -> Managed Identity -> CLI -> etc.)
        _azure_credential = DefaultAzureCredential(
            # Exclude problematic credential types in local development
            exclude_managed_identity_credential=False,  # Allow managed identity but with timeout
            managed_identity_credential_timeout=30,     # 30 second timeout instead of 2 minutes
            exclude_powershell_credential=True,         # Exclude PowerShell for reliability
        )
        logger.info("Azure credential refreshed with DefaultAzureCredential")
    except Exception as e:
        logger.error(f"Credential refresh failed: {e}")
        raise ClientAuthenticationError(f"Failed to refresh Azure credentials: {e}")

    _last_refresh_time = time.time()
    logger.debug(f"Azure credential refreshed at {_last_refresh_time}")


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
    global _azure_credential, _last_refresh_time
    _azure_credential = None
    _last_refresh_time = 0
    logger.info("Azure credential reset")


def get_session_metadata() -> dict:
    """
    Get current session metadata for monitoring and debugging.

    Returns:
        dict: Session information including refresh time and credential type
    """
    global _azure_credential, _last_refresh_time

    return {
        "credential_initialized": _azure_credential is not None,
        "last_refresh_time": _last_refresh_time,
        "time_since_refresh_minutes": (
            (time.time() - _last_refresh_time) / 60 if _last_refresh_time > 0 else 0
        ),
        "credential_type": (
            type(_azure_credential).__name__ if _azure_credential else None
        ),
        "refresh_threshold_minutes": _refresh_threshold_minutes,
    }


def configure_refresh_threshold(minutes: int):
    """
    Configure the credential refresh threshold.

    Args:
        minutes: Number of minutes before credential expiry to refresh
    """
    global _refresh_threshold_minutes
    _refresh_threshold_minutes = minutes
    logger.info(f"Azure credential refresh threshold set to {minutes} minutes")


def force_credential_refresh():
    """
    Force immediate credential refresh regardless of timing.

    Useful for error recovery scenarios.
    """
    logger.info("Forcing Azure credential refresh")
    _refresh_credential()
