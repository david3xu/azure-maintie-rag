"""
Configuration enforcement system to prevent hardcoded values.

This module implements runtime validation that ensures all configuration
comes from workflow-generated intelligence, not hardcoded fallbacks.
"""

from typing import Any, Dict, Optional
import os
import inspect
import traceback
from datetime import datetime


class ConfigurationEnforcementError(Exception):
    """Raised when hardcoded values are detected"""
    pass


class AntiHardcodingEnforcer:
    """Prevents hardcoded values from being used in production"""
    
    def __init__(self):
        self.is_development = os.getenv("ENVIRONMENT", "production").lower() == "development"
        self.violations_log = []
    
    def validate_configuration_source(self, config_key: str, value: Any, source: str) -> Any:
        """Validate that configuration comes from proper workflow sources"""
        
        # Check if value comes from hardcoded source
        if self._is_hardcoded_source(source):
            violation = {
                "key": config_key,
                "value": value,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "stack_trace": self._get_caller_info()
            }
            
            self.violations_log.append(violation)
            
            if not self.is_development:
                raise ConfigurationEnforcementError(
                    f"HARDCODED VALUE DETECTED: {config_key}={value} from {source}. "
                    f"System configured for data-driven intelligence, not hardcoded fallbacks. "
                    f"Run Config-Extraction workflow first or set ENVIRONMENT=development."
                )
            else:
                print(f"⚠️  DEVELOPMENT WARNING: Using hardcoded {config_key}={value}")
                
        return value
    
    def _is_hardcoded_source(self, source: str) -> bool:
        """Detect hardcoded value sources"""
        hardcoded_indicators = [
            "default_value",
            "placeholder", 
            "mock_implementation",
            "hardcoded",
            "fallback",
            "TODO",
            "__file__"  # Values defined in same file
        ]
        return any(indicator in source.lower() for indicator in hardcoded_indicators)
    
    def _get_caller_info(self) -> str:
        """Get information about where the configuration was used"""
        try:
            # Get the call stack
            stack = traceback.extract_stack()
            # Find the first non-enforcement frame
            for frame in reversed(stack[:-2]):  # Skip current and validate_configuration_source
                if 'config_enforcement' not in frame.filename:
                    return f"{frame.filename}:{frame.lineno} in {frame.name}"
            return "Unknown caller"
        except Exception:
            return "Stack trace unavailable"
    
    def get_violation_report(self) -> Dict[str, Any]:
        """Get a report of all configuration violations"""
        return {
            "total_violations": len(self.violations_log),
            "violations": self.violations_log,
            "environment": "development" if self.is_development else "production",
            "report_generated": datetime.now().isoformat()
        }
    
    def clear_violations(self) -> int:
        """Clear the violations log and return count of cleared items"""
        count = len(self.violations_log)
        self.violations_log.clear()
        return count


# Global enforcer instance for easy access
_global_enforcer = AntiHardcodingEnforcer()


def validate_config(config_key: str, value: Any, source: str = "unknown") -> Any:
    """
    Global function to validate configuration values.
    
    This is a convenience function that uses the global enforcer instance.
    Use this in existing code to quickly add validation.
    
    Args:
        config_key: Name of the configuration parameter
        value: The configuration value
        source: Description of where the value comes from
        
    Returns:
        The validated value (same as input if valid)
        
    Raises:
        ConfigurationEnforcementError: If hardcoded value detected in production
    """
    return _global_enforcer.validate_configuration_source(config_key, value, source)


def get_enforcement_report() -> Dict[str, Any]:
    """Get a report of all configuration violations from global enforcer"""
    return _global_enforcer.get_violation_report()


def clear_enforcement_violations() -> int:
    """Clear violations from global enforcer"""
    return _global_enforcer.clear_violations()