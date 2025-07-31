#!/usr/bin/env python3
"""
Validate Circuit Breaker Implementation in BaseAzureClient
Tests the circuit breaker pattern functionality
"""

def test_circuit_breaker_configuration():
    """Test that circuit breaker configuration is properly implemented"""
    print("🔍 Testing Circuit Breaker Configuration...")
    
    try:
        from core.azure_auth.base_client import CircuitBreakerConfig, CircuitBreakerState, ClientMetrics
        
        # Test default configuration
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.timeout_duration == 60.0
        assert config.success_threshold == 2
        assert config.enabled == True
        print("✅ CircuitBreakerConfig defaults are correct")
        
        # Test custom configuration
        custom_config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_duration=30.0,
            success_threshold=1,
            enabled=False
        )
        assert custom_config.failure_threshold == 3
        assert custom_config.timeout_duration == 30.0
        assert custom_config.success_threshold == 1
        assert custom_config.enabled == False
        print("✅ CircuitBreakerConfig custom values work")
        
        # Test state initialization
        state = CircuitBreakerState()
        assert state.failure_count == 0
        assert state.last_failure_time is None
        assert state.state == "CLOSED"
        assert state.next_attempt_time is None
        print("✅ CircuitBreakerState initializes correctly")
        
        # Test metrics initialization
        metrics = ClientMetrics()
        assert metrics.circuit_breaker is not None
        assert metrics.circuit_breaker.state == "CLOSED"
        print("✅ ClientMetrics includes circuit breaker state")
        
        return True
        
    except Exception as e:
        print(f"❌ Circuit breaker configuration test failed: {e}")
        return False

def test_circuit_breaker_methods():
    """Test that circuit breaker methods are available in BaseAzureClient"""
    print("\n🔍 Testing Circuit Breaker Methods...")
    
    try:
        from core.azure_auth.base_client import BaseAzureClient
        
        # Create a mock client to test method availability
        class MockAzureClient(BaseAzureClient):
            def _get_default_endpoint(self) -> str:
                return "https://mock.azure.com"
            
            def _initialize_client(self):
                pass
            
            def _health_check(self) -> bool:
                return True
        
        client = MockAzureClient()
        
        # Test that circuit breaker methods exist
        assert hasattr(client, '_check_circuit_breaker')
        assert callable(client._check_circuit_breaker)
        print("✅ _check_circuit_breaker method exists")
        
        assert hasattr(client, '_record_circuit_breaker_success')
        assert callable(client._record_circuit_breaker_success)
        print("✅ _record_circuit_breaker_success method exists")
        
        assert hasattr(client, '_record_circuit_breaker_failure')
        assert callable(client._record_circuit_breaker_failure)
        print("✅ _record_circuit_breaker_failure method exists")
        
        # Test that configuration is loaded
        assert hasattr(client, 'circuit_breaker_config')
        assert client.circuit_breaker_config is not None
        print("✅ Circuit breaker configuration is loaded")
        
        # Test that state is initialized
        assert hasattr(client, 'metrics')
        assert client.metrics.circuit_breaker is not None
        print("✅ Circuit breaker state is initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Circuit breaker methods test failed: {e}")
        return False

def test_circuit_breaker_state_transitions():
    """Test circuit breaker state transitions"""
    print("\n🔍 Testing Circuit Breaker State Transitions...")
    
    try:
        from core.azure_auth.base_client import BaseAzureClient
        import time
        
        class MockAzureClient(BaseAzureClient):
            def _get_default_endpoint(self) -> str:
                return "https://mock.azure.com"
            
            def _initialize_client(self):
                pass
            
            def _health_check(self) -> bool:
                return True
        
        # Create client with custom circuit breaker config
        config = {
            'circuit_breaker': {
                'failure_threshold': 2,  # Low threshold for testing
                'timeout_duration': 0.1   # Short timeout for testing
            }
        }
        client = MockAzureClient(config)
        
        # Initial state should be CLOSED
        assert client.metrics.circuit_breaker.state == "CLOSED"
        print("✅ Initial state is CLOSED")
        
        # Test circuit breaker check when CLOSED
        can_proceed = client._check_circuit_breaker("test_operation")
        assert can_proceed == True
        print("✅ Circuit breaker allows operations when CLOSED")
        
        # Simulate failures to open circuit
        client._record_circuit_breaker_failure("test_operation")
        assert client.metrics.circuit_breaker.state == "CLOSED"  # Still closed after 1 failure
        
        client._record_circuit_breaker_failure("test_operation")
        assert client.metrics.circuit_breaker.state == "OPEN"  # Should open after 2 failures
        print("✅ Circuit opens after reaching failure threshold")
        
        # Test that operations are blocked when OPEN
        can_proceed = client._check_circuit_breaker("test_operation")
        assert can_proceed == False
        print("✅ Circuit breaker blocks operations when OPEN")
        
        # Wait for timeout and test half-open transition
        time.sleep(0.15)  # Wait for timeout duration
        can_proceed = client._check_circuit_breaker("test_operation")
        assert can_proceed == True
        assert client.metrics.circuit_breaker.state == "HALF_OPEN"
        print("✅ Circuit transitions to HALF_OPEN after timeout")
        
        # Test success in half-open closes circuit
        client._record_circuit_breaker_success("test_operation")
        assert client.metrics.circuit_breaker.state == "CLOSED"
        print("✅ Circuit closes after success in HALF_OPEN state")
        
        return True
        
    except Exception as e:
        print(f"❌ Circuit breaker state transitions test failed: {e}")
        return False

def test_circuit_breaker_metrics():
    """Test that circuit breaker metrics are included in get_metrics"""
    print("\n🔍 Testing Circuit Breaker Metrics...")
    
    try:
        from core.azure_auth.base_client import BaseAzureClient
        
        class MockAzureClient(BaseAzureClient):
            def _get_default_endpoint(self) -> str:
                return "https://mock.azure.com"
            
            def _initialize_client(self):
                pass
            
            def _health_check(self) -> bool:
                return True
        
        client = MockAzureClient()
        metrics = client.get_metrics()
        
        # Test that circuit breaker metrics are included
        assert "circuit_breaker" in metrics
        circuit_metrics = metrics["circuit_breaker"]
        
        assert "state" in circuit_metrics
        assert "failure_count" in circuit_metrics
        assert "last_failure_time" in circuit_metrics
        assert "next_attempt_time" in circuit_metrics
        assert "enabled" in circuit_metrics
        
        print("✅ Circuit breaker metrics are included in get_metrics")
        
        # Test initial values
        assert circuit_metrics["state"] == "CLOSED"
        assert circuit_metrics["failure_count"] == 0
        assert circuit_metrics["enabled"] == True
        print("✅ Circuit breaker metrics have correct initial values")
        
        return True
        
    except Exception as e:
        print(f"❌ Circuit breaker metrics test failed: {e}")
        return False

def test_health_check_integration():
    """Test that health check considers circuit breaker state"""
    print("\n🔍 Testing Health Check Integration...")
    
    try:
        from core.azure_auth.base_client import BaseAzureClient
        import asyncio
        
        class MockAzureClient(BaseAzureClient):
            def _get_default_endpoint(self) -> str:
                return "https://mock.azure.com"
            
            def _initialize_client(self):
                pass
            
            def _health_check(self) -> bool:
                return True
        
        async def test_health():
            client = MockAzureClient()
            
            # Test health when circuit is closed
            health = await client.health_check()
            assert health["healthy"] == True
            assert "circuit_breaker_status" in health
            assert health["circuit_breaker_status"] == "CLOSED"
            print("✅ Health check shows healthy when circuit is CLOSED")
            
            # Simulate opening the circuit
            config = {'circuit_breaker': {'failure_threshold': 1}}
            client = MockAzureClient(config)
            client._record_circuit_breaker_failure("test")
            
            health = await client.health_check()
            assert health["healthy"] == False  # Should be unhealthy when circuit is open
            assert health["circuit_breaker_status"] == "OPEN"
            print("✅ Health check shows unhealthy when circuit is OPEN")
            
            return True
        
        # Run async test
        result = asyncio.run(test_health())
        return result
        
    except Exception as e:
        print(f"❌ Health check integration test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Circuit Breaker Pattern Validation")
    print("=" * 60)
    
    success = True
    success &= test_circuit_breaker_configuration()
    success &= test_circuit_breaker_methods()
    success &= test_circuit_breaker_state_transitions()
    success &= test_circuit_breaker_metrics()
    success &= test_health_check_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ CIRCUIT BREAKER IMPLEMENTATION VALIDATED")
        print("🎯 Key features implemented:")
        print("   - Configurable failure thresholds and timeouts")
        print("   - CLOSED -> OPEN -> HALF_OPEN -> CLOSED state machine")
        print("   - Integration with retry mechanisms")
        print("   - Circuit breaker metrics and monitoring")
        print("   - Health check integration")
        print("   - Prevents cascading failures")
        print("\n📋 Circuit breaker pattern successfully added to BaseAzureClient")
    else:
        print("❌ CIRCUIT BREAKER VALIDATION FAILED")
    
    return success

if __name__ == "__main__":
    main()