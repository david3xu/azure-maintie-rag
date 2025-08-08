"""
Test utilities for Azure Universal RAG system
Provides common error handling patterns for Azure service tests
"""

import pytest
import functools
from typing import Callable, Any


def handle_azure_service_errors(test_func: Callable) -> Callable:
    """
    Decorator that provides consistent Azure service error handling for tests.
    
    This decorator wraps test functions to provide graceful error handling for:
    - Authentication issues
    - Network connectivity problems  
    - Azure service configuration problems
    - Rate limiting
    - Resource not found errors
    
    Usage:
        @handle_azure_service_errors
        async def test_azure_service(self):
            # Your test code here
            pass
    """
    
    @functools.wraps(test_func)
    async def wrapper(*args, **kwargs):
        try:
            return await test_func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            
            # Authentication issues
            if any(auth_error in error_msg for auth_error in [
                'authentication', 'credential', 'unauthorized', 'forbidden',
                'invalid_api_key', 'access_denied', 'token', 'login required',
                'az login', 'no subscription', '401', '403'
            ]):
                pytest.skip(f"Azure authentication issue - check credentials: {e}")
            
            # Network/connectivity issues
            if any(network_error in error_msg for network_error in [
                'connection', 'timeout', 'network', 'dns', 'socket', 'unreachable',
                'connection reset', 'connection refused'
            ]):
                pytest.skip(f"Network connectivity issue - check Azure services: {e}")
            
            # Resource configuration issues
            if any(config_error in error_msg for config_error in [
                'not found', '404', 'resource not found', 'deployment not found',
                'index not found', 'container not found', 'endpoint not found'
            ]):
                pytest.skip(f"Azure resource configuration issue: {e}")
            
            # Rate limiting
            if any(rate_error in error_msg for rate_error in [
                'rate limit', 'quota', 'throttl', '429', 'too many requests'
            ]):
                pytest.skip(f"Azure service rate limit - try again later: {e}")
            
            # For any other errors, re-raise to show in test output
            raise
    
    # Handle synchronous test functions too
    if not hasattr(test_func, '__code__') or not any('await' in line for line in test_func.__code__.co_names):
        @functools.wraps(test_func)
        def sync_wrapper(*args, **kwargs):
            try:
                return test_func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                
                # Same error handling logic for sync functions
                if any(auth_error in error_msg for auth_error in [
                    'authentication', 'credential', 'unauthorized', 'forbidden',
                    'invalid_api_key', 'access_denied', 'token', 'login required',
                    'az login', 'no subscription', '401', '403'
                ]):
                    pytest.skip(f"Azure authentication issue - check credentials: {e}")
                
                if any(network_error in error_msg for network_error in [
                    'connection', 'timeout', 'network', 'dns', 'socket', 'unreachable',
                    'connection reset', 'connection refused'
                ]):
                    pytest.skip(f"Network connectivity issue - check Azure services: {e}")
                
                if any(config_error in error_msg for config_error in [
                    'not found', '404', 'resource not found', 'deployment not found',
                    'index not found', 'container not found', 'endpoint not found'
                ]):
                    pytest.skip(f"Azure resource configuration issue: {e}")
                
                if any(rate_error in error_msg for rate_error in [
                    'rate limit', 'quota', 'throttl', '429', 'too many requests'
                ]):
                    pytest.skip(f"Azure service rate limit - try again later: {e}")
                
                raise
        
        return sync_wrapper
    
    return wrapper


def print_azure_diagnostic_info(service_name: str, **kwargs):
    """
    Print diagnostic information for Azure service tests.
    
    Args:
        service_name: Name of the Azure service being tested
        **kwargs: Additional diagnostic information to display
    """
    print(f"\n🔍 {service_name} Diagnostic Information:")
    print("-" * 50)
    
    for key, value in kwargs.items():
        if value:
            # Hide sensitive values
            if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'token', 'password']):
                display_value = f"[SET - {len(str(value))} chars]"
            elif len(str(value)) > 80:
                display_value = f"{str(value)[:50]}...{str(value)[-15:]}"
            else:
                display_value = str(value)
            print(f"   {key}: {display_value}")
        else:
            print(f"   {key}: NOT SET")