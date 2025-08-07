"""Azure Session Management with Auto-Refresh"""

import logging
import time
from typing import Any, Dict, Optional

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

logger = logging.getLogger(__name__)


class AzureSessionManager:
    """Enterprise Azure session management with auto-refresh"""

    def __init__(self, refresh_threshold_minutes: int = 50):
        self.refresh_threshold_minutes = refresh_threshold_minutes
        self.credential = None
        self.last_refresh_time = 0
        self.session_metadata = {}

    def get_credential(self) -> DefaultAzureCredential:
        """Get credential with automatic refresh"""
        current_time = time.time()
        time_since_refresh = (current_time - self.last_refresh_time) / 60

        if not self.credential or time_since_refresh > self.refresh_threshold_minutes:
            self._refresh_credential()

        return self.credential

    def _refresh_credential(self):
        """Refresh Azure credential with fallback patterns"""
        try:
            # Try managed identity first (for Azure environments)
            self.credential = ManagedIdentityCredential()
            logger.info("Azure session refreshed with Managed Identity")
        except Exception:
            try:
                # Fallback to default credential chain
                self.credential = DefaultAzureCredential()
                logger.info("Azure session refreshed with Default Credential")
            except Exception as e:
                logger.error(f"Credential refresh failed: {e}")
                raise ClientAuthenticationError(
                    f"Failed to refresh Azure credentials: {e}"
                )

        self.last_refresh_time = time.time()
        self.session_metadata = {
            "refreshed_at": self.last_refresh_time,
            "credential_type": type(self.credential).__name__,
        }
