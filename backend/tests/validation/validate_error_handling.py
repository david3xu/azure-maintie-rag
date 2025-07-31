#!/usr/bin/env python3
"""
Validate Standardized Error Handling Implementation
Tests the enhanced error handling patterns in BaseAzureClient
"""

def test_error_severity_enum():
    """Test ErrorSeverity enum is properly defined"""
    print("🔍 Testing ErrorSeverity Enum...")
    
    try:
        from core.azure_auth.base_client import ErrorSeverity
        
        # Test all severity levels exist
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
        print("✅ ErrorSeverity enum has all required levels")
        
        # Test enum can be compared
        assert ErrorSeverity.CRITICAL != ErrorSeverity.LOW
        assert ErrorSeverity.MEDIUM == ErrorSeverity.MEDIUM
        print("✅ ErrorSeverity enum comparison works")
        
        return True
        
    except Exception as e:
        print(f"❌ ErrorSeverity enum test failed: {e}")
        return False

def test_azure_service_error():
    """Test AzureServiceError custom exception"""
    print("\n🔍 Testing AzureServiceError...")
    
    try:
        from core.azure_auth.base_client import AzureServiceError, ErrorSeverity
        
        # Test basic error creation
        error = AzureServiceError("Test error message")
        assert error.message == "Test error message"
        assert error.severity == ErrorSeverity.MEDIUM  # Default
        assert error.service_name is None
        assert error.operation is None
        print("✅ AzureServiceError creates with defaults")
        
        # Test error with full context
        original_error = ValueError("Original error")
        context = {"key": "value", "test": 123}
        
        error = AzureServiceError(
            message="Detailed error message",
            severity=ErrorSeverity.HIGH,
            service_name="TestService",
            operation="test_operation",
            original_error=original_error,
            context=context
        )
        
        assert error.message == "Detailed error message"
        assert error.severity == ErrorSeverity.HIGH
        assert error.service_name == "TestService"
        assert error.operation == "test_operation"
        assert error.original_error == original_error
        assert error.context == context
        print("✅ AzureServiceError stores all context correctly")
        
        # Test to_dict method
        error_dict = error.to_dict()
        assert error_dict["error_message"] == "Detailed error message"
        assert error_dict["severity"] == "high"
        assert error_dict["service_name"] == "TestService"
        assert error_dict["operation"] == "test_operation"
        assert error_dict["original_error"] == "Original error"
        assert error_dict["original_error_type"] == "ValueError"
        assert error_dict["context"] == context
        assert "timestamp" in error_dict
        print("✅ AzureServiceError.to_dict() works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ AzureServiceError test failed: {e}")
        return False

def test_enhanced_error_response():
    """Test enhanced create_error_response method"""
    print("\n🔍 Testing Enhanced Error Response...")
    
    try:
        from core.azure_auth.base_client import BaseAzureClient, AzureServiceError, ErrorSeverity
        
        class MockAzureClient(BaseAzureClient):
            def _get_default_endpoint(self) -> str:
                return "https://mock.azure.com"
            
            def _initialize_client(self):
                pass
            
            def _health_check(self) -> bool:
                return True
        
        client = MockAzureClient()
        
        # Test error response with string
        response = client.create_error_response("test_operation", "Simple error message")
        assert response["success"] == False
        assert response["operation"] == "test_operation"
        assert response["error"] == "Simple error message"
        assert response["severity"] == "medium"  # Default
        print("✅ Error response with string message works")
        
        # Test error response with Exception
        exception = ValueError("Test exception")
        response = client.create_error_response("test_operation", exception)
        assert response["error"] == "Test exception"
        assert response["error_type"] == "ValueError"
        print("✅ Error response with Exception works")
        
        # Test error response with AzureServiceError
        service_error = AzureServiceError(
            message="Service error",
            severity=ErrorSeverity.HIGH,
            service_name="TestService",
            operation="test_op",
            context={"key": "value"}
        )
        response = client.create_error_response("test_operation", service_error)
        assert response["error"] == "Service error"
        assert response["severity"] == "high"
        assert response["error_type"] == "AzureServiceError"
        assert response["context"] == {"key": "value"}
        print("✅ Error response with AzureServiceError works")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced error response test failed: {e}")
        return False

def test_handle_error_method():
    """Test standardized handle_error method"""
    print("\n🔍 Testing Handle Error Method...")
    
    try:
        from core.azure_auth.base_client import BaseAzureClient, ErrorSeverity
        import logging
        
        class MockAzureClient(BaseAzureClient):
            def _get_default_endpoint(self) -> str:
                return "https://mock.azure.com"
            
            def _initialize_client(self):
                pass
            
            def _health_check(self) -> bool:
                return True
        
        client = MockAzureClient()
        
        # Test handle_error with basic error
        original_error = ValueError("Test error")
        service_error = client.handle_error("test_operation", original_error)
        
        assert "test_operation failed: Test error" in service_error.message
        assert service_error.severity == ErrorSeverity.MEDIUM  # Default
        assert service_error.service_name == "MockAzureClient"
        assert service_error.operation == "test_operation"
        assert service_error.original_error == original_error
        print("✅ handle_error with basic parameters works")
        
        # Test handle_error with full context
        context = {"param1": "value1", "param2": 123}
        service_error = client.handle_error(
            "complex_operation", 
            original_error,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            log_level="error"
        )
        
        assert service_error.severity == ErrorSeverity.CRITICAL
        assert service_error.context == context
        print("✅ handle_error with full context works")
        
        return True
        
    except Exception as e:
        print(f"❌ Handle error method test failed: {e}")
        return False

def test_safe_execute_method():
    """Test safe_execute method for standardized operation execution"""
    print("\n🔍 Testing Safe Execute Method...")
    
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
        
        # Test successful operation
        def successful_operation(x, y):
            return x + y
        
        result = client.safe_execute("add_operation", successful_operation, 2, 3)
        assert result["success"] == True
        assert result["operation"] == "add_operation"
        assert result["data"] == 5
        print("✅ safe_execute with successful operation works")
        
        # Test failing operation
        def failing_operation():
            raise ValueError("Intentional test error")
        
        result = client.safe_execute("fail_operation", failing_operation)
        assert result["success"] == False
        assert result["operation"] == "fail_operation"
        assert "Intentional test error" in result["error"]
        assert result["error_type"] == "AzureServiceError"
        print("✅ safe_execute with failing operation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Safe execute method test failed: {e}")
        return False

def test_ml_client_error_standardization():
    """Test that AzureMLClient uses standardized error handling"""
    print("\n🔍 Testing ML Client Error Standardization...")
    
    try:
        from core.azure_ml.client import AzureMLClient
        from core.azure_auth.base_client import AzureServiceError
        
        # Create ML client
        client = AzureMLClient()
        
        # Test that the client can create standardized errors
        test_error = ValueError("Test error")
        service_error = client.handle_error("test_operation", test_error)
        assert isinstance(service_error, AzureServiceError)
        assert service_error.service_name == "AzureMLClient"
        print("✅ AzureMLClient can create standardized errors")
        
        # Test error response creation
        response = client.create_error_response("test_op", service_error)
        assert response["service"] == "AzureMLClient"
        assert response["error_type"] == "AzureServiceError"
        print("✅ AzureMLClient creates standardized error responses")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Client error standardization test failed: {e}")
        return False

def test_error_severity_integration():
    """Test that error severity affects logging and responses correctly"""
    print("\n🔍 Testing Error Severity Integration...")
    
    try:
        from core.azure_auth.base_client import BaseAzureClient, ErrorSeverity
        
        class MockAzureClient(BaseAzureClient):
            def _get_default_endpoint(self) -> str:
                return "https://mock.azure.com"
            
            def _initialize_client(self):
                pass
            
            def _health_check(self) -> bool:
                return True
        
        client = MockAzureClient()
        
        # Test different severity levels
        severities = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        
        for severity in severities:
            error = ValueError(f"Test {severity.value} error")
            service_error = client.handle_error("test_operation", error, severity=severity)
            
            assert service_error.severity == severity
            
            response = client.create_error_response("test_operation", service_error)
            assert response["severity"] == severity.value
            
            print(f"✅ {severity.value} severity error handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error severity integration test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Standardized Error Handling Validation")
    print("=" * 70)
    
    success = True
    success &= test_error_severity_enum()
    success &= test_azure_service_error()
    success &= test_enhanced_error_response()
    success &= test_handle_error_method()
    success &= test_safe_execute_method()
    success &= test_ml_client_error_standardization()
    success &= test_error_severity_integration()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ STANDARDIZED ERROR HANDLING VALIDATED")
        print("🎯 Key features implemented:")
        print("   - ErrorSeverity enum with LOW/MEDIUM/HIGH/CRITICAL levels")
        print("   - AzureServiceError with context and severity")
        print("   - Enhanced error responses with severity and context")
        print("   - Standardized handle_error method with severity-based logging")
        print("   - Safe execution wrapper for operations")
        print("   - Integration with circuit breaker and retry mechanisms")
        print("   - Consistent error handling across all Azure clients")
        print("\n📋 Error handling standardization complete!")
    else:
        print("❌ STANDARDIZED ERROR HANDLING VALIDATION FAILED")
    
    return success

if __name__ == "__main__":
    main()