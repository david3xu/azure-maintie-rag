"""
Simplified Configuration Validation - PydanticAI Built-in Patterns

Replaces complex enforcement system with simple validation using PydanticAI Field constraints.
"""

from typing import Any, Dict
import os
from pydantic import BaseModel, Field


class ConfigurationEnforcementError(Exception):
    """Raised when configuration validation fails"""

    pass


# Simple validation using PydanticAI built-in patterns
def validate_config(config_key: str, value: Any, source: str = "unknown") -> Any:
    """
    Simplified validation using PydanticAI built-in patterns.

    Returns value directly - validation handled by PydanticAI Field constraints.
    """
    # In development environment, just pass through
    if os.getenv("ENVIRONMENT", "production").lower() == "development":
        return value

    # In production, basic validation - most validation now handled by PydanticAI Field constraints
    if value is None:
        raise ConfigurationEnforcementError(
            f"Configuration {config_key} cannot be None"
        )

    return value


# Simplified status functions - replaced complex tracking with basic status
def get_enforcement_report() -> Dict[str, Any]:
    """Simple enforcement report - complex tracking replaced with PydanticAI validation"""
    return {
        "status": "simplified_validation",
        "validation_method": "pydantic_field_constraints",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "note": "Complex validation replaced with PydanticAI built-in Field constraints",
    }


def clear_enforcement_violations() -> int:
    """No violations to clear - validation simplified"""
    return 0


class AntiHardcodingEnforcer:
    """Simplified enforcer - replaced complex system with PydanticAI built-in validation"""

    def __init__(self):
        self.simplified = True

    def validate_config(self, key: str, value: Any, source: str = "unknown") -> Any:
        """Delegate to simplified validation function"""
        return validate_config(key, value, source)

    def get_report(self) -> Dict[str, Any]:
        """Get simplified enforcement report"""
        return get_enforcement_report()
