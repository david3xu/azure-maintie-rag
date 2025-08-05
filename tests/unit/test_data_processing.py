"""
Data Processing Unit Tests - CODING_STANDARDS Compliant
Tests data transformation utilities without Azure dependencies.
"""

import pytest
import json
from typing import Dict, List, Any


class TestDataTransformationUtils:
    """Test data transformation and utility functions"""
    
    def test_text_preprocessing(self):
        """Test text preprocessing utilities"""
        
        # Test cases for text cleaning
        test_cases = [
            {
                "input": "  Hello World  ",
                "expected": "Hello World",
                "operation": "trim_whitespace"
            },
            {
                "input": "Text\nwith\nnewlines",
                "expected": "Text with newlines", 
                "operation": "normalize_newlines"
            },
            {
                "input": "Text    with    multiple    spaces",
                "expected": "Text with multiple spaces",
                "operation": "normalize_spaces"
            },
            {
                "input": "Mixed\t\r\n\t whitespace",
                "expected": "Mixed whitespace",
                "operation": "normalize_all_whitespace"
            }
        ]
        
        for test_case in test_cases:
            # Simple preprocessing implementation
            text = test_case["input"]
            
            # Apply transformations
            text = text.strip()  # Trim whitespace
            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')  # Normalize newlines/tabs
            text = ' '.join(text.split())  # Normalize multiple spaces
            
            assert text == test_case["expected"], f"Failed {test_case['operation']}: '{text}' != '{test_case['expected']}'"
        
        print("✅ Text preprocessing validation passed")
    
    def test_data_structure_validation(self):
        """Test data structure validation utilities"""
        
        # Test valid data structures
        valid_structures = [
            {
                "data": {"query": "test", "limit": 10},
                "required_fields": ["query"],
                "valid": True
            },
            {
                "data": {"results": [{"content": "test"}], "count": 1},
                "required_fields": ["results"],
                "valid": True
            },
            {
                "data": {"entities": [], "relationships": []},
                "required_fields": ["entities", "relationships"],
                "valid": True
            }
        ]
        
        # Test invalid data structures
        invalid_structures = [
            {
                "data": {"limit": 10},  # Missing required query
                "required_fields": ["query"],
                "valid": False
            },
            {
                "data": None,
                "required_fields": ["query"],
                "valid": False
            },
            {
                "data": {"query": ""},  # Empty required field
                "required_fields": ["query"],
                "valid": False,
                "check_empty": True
            }
        ]
        
        def validate_data_structure(data: Any, required_fields: List[str], check_empty: bool = False) -> bool:
            """Simple data structure validation"""
            if not isinstance(data, dict):
                return False
            
            for field in required_fields:
                if field not in data:
                    return False
                if check_empty and not data[field]:
                    return False
            
            return True
        
        # Test valid structures
        for test in valid_structures:
            result = validate_data_structure(
                test["data"], 
                test["required_fields"],
                test.get("check_empty", False)
            )
            assert result == test["valid"], f"Valid structure test failed: {test}"
        
        # Test invalid structures
        for test in invalid_structures:
            result = validate_data_structure(
                test["data"], 
                test["required_fields"],
                test.get("check_empty", False)
            )
            assert result == test["valid"], f"Invalid structure test failed: {test}"
        
        print("✅ Data structure validation passed")
    
    def test_json_serialization_handling(self):
        """Test JSON serialization and deserialization handling"""
        
        # Test data that should serialize/deserialize correctly
        test_data = [
            {"simple": "string", "number": 42, "boolean": True},
            {"nested": {"data": [1, 2, 3]}},
            {"entities": [{"text": "Python", "type": "language"}]},
            {"relationships": [{"source": "A", "target": "B", "relation": "uses"}]}
        ]
        
        for data in test_data:
            # Test serialization
            json_str = json.dumps(data)
            assert isinstance(json_str, str)
            assert len(json_str) > 0
            
            # Test deserialization
            restored_data = json.loads(json_str)
            assert restored_data == data
        
        # Test error handling for non-serializable data
        class NonSerializable:
            pass
        
        try:
            json.dumps({"object": NonSerializable()})
            pytest.fail("Should have raised TypeError for non-serializable data")
        except TypeError:
            pass  # Expected behavior
        
        print("✅ JSON serialization handling passed")
    
    def test_list_processing_utilities(self):
        """Test list processing and filtering utilities"""
        
        # Sample data for testing
        sample_results = [
            {"content": "Result 1", "score": 0.9, "domain": "programming"},
            {"content": "Result 2", "score": 0.8, "domain": "programming"},
            {"content": "Result 3", "score": 0.7, "domain": "medical"},
            {"content": "Result 4", "score": 0.6, "domain": "legal"},
            {"content": "Result 5", "score": 0.5, "domain": "medical"}
        ]
        
        # Test filtering by score threshold
        def filter_by_score(results: List[Dict], threshold: float) -> List[Dict]:
            return [r for r in results if r.get("score", 0) >= threshold]
        
        high_score_results = filter_by_score(sample_results, 0.75)
        assert len(high_score_results) == 2  # Scores 0.9, 0.8 (0.7 < 0.75)
        assert all(r["score"] >= 0.75 for r in high_score_results)
        
        # Test filtering by domain
        def filter_by_domain(results: List[Dict], domain: str) -> List[Dict]:
            return [r for r in results if r.get("domain") == domain]
        
        programming_results = filter_by_domain(sample_results, "programming")
        assert len(programming_results) == 2
        assert all(r["domain"] == "programming" for r in programming_results)
        
        # Test sorting by score (descending)
        def sort_by_score(results: List[Dict], descending: bool = True) -> List[Dict]:
            return sorted(results, key=lambda x: x.get("score", 0), reverse=descending)
        
        sorted_results = sort_by_score(sample_results)
        assert sorted_results[0]["score"] == 0.9  # Highest first
        assert sorted_results[-1]["score"] == 0.5  # Lowest last
        
        # Test limiting results
        def limit_results(results: List[Dict], limit: int) -> List[Dict]:
            return results[:limit] if limit > 0 else results
        
        limited_results = limit_results(sample_results, 3)
        assert len(limited_results) == 3
        
        print("✅ List processing utilities validation passed")
    
    def test_data_aggregation_utilities(self):
        """Test data aggregation and statistical utilities"""
        
        # Sample data for aggregation
        performance_data = [
            {"operation": "search", "duration": 1.2, "success": True},
            {"operation": "search", "duration": 1.5, "success": True},
            {"operation": "extraction", "duration": 2.3, "success": True},
            {"operation": "extraction", "duration": 2.1, "success": False},
            {"operation": "search", "duration": 0.9, "success": True}
        ]
        
        # Test calculating averages
        def calculate_average_duration(data: List[Dict]) -> float:
            durations = [d["duration"] for d in data]
            return sum(durations) / len(durations) if durations else 0.0
        
        avg_duration = calculate_average_duration(performance_data)
        expected_avg = (1.2 + 1.5 + 2.3 + 2.1 + 0.9) / 5
        assert abs(avg_duration - expected_avg) < 0.01
        
        # Test calculating success rates
        def calculate_success_rate(data: List[Dict]) -> float:
            if not data:
                return 0.0
            successful = sum(1 for d in data if d.get("success", False))
            return successful / len(data)
        
        success_rate = calculate_success_rate(performance_data)
        assert abs(success_rate - 0.8) < 0.01  # 4 out of 5 successful
        
        # Test grouping by operation
        def group_by_operation(data: List[Dict]) -> Dict[str, List[Dict]]:
            groups = {}
            for item in data:
                operation = item.get("operation", "unknown")
                if operation not in groups:
                    groups[operation] = []
                groups[operation].append(item)
            return groups
        
        grouped = group_by_operation(performance_data)
        assert "search" in grouped
        assert "extraction" in grouped
        assert len(grouped["search"]) == 3
        assert len(grouped["extraction"]) == 2
        
        print("✅ Data aggregation utilities validation passed")


class TestDataValidationUtils:
    """Test data validation and error handling utilities"""
    
    def test_input_sanitization(self):
        """Test input sanitization utilities"""
        
        # Test query sanitization
        def sanitize_query(query: str) -> str:
            if not isinstance(query, str):
                return ""
            
            # Remove excessive whitespace
            query = ' '.join(query.split())
            
            # Limit length (example limit)
            max_length = 1000
            if len(query) > max_length:
                query = query[:max_length]
            
            return query.strip()
        
        test_cases = [
            ("  normal query  ", "normal query"),
            ("query\nwith\nnewlines", "query with newlines"),
            ("a" * 1500, "a" * 1000),  # Truncated to limit
            ("", ""),
            (None, "")  # Handle None input
        ]
        
        for input_val, expected in test_cases:
            try:
                result = sanitize_query(input_val)
                assert result == expected, f"Sanitization failed: '{result}' != '{expected}'"
            except Exception as e:
                if expected == "":  # Expected to handle gracefully
                    assert True
                else:
                    pytest.fail(f"Unexpected error for input '{input_val}': {e}")
        
        print("✅ Input sanitization validation passed")
    
    def test_range_validation(self):
        """Test numeric range validation utilities"""
        
        def validate_range(value: Any, min_val: float, max_val: float, default: float = None) -> float:
            """Validate numeric value is within range"""
            try:
                numeric_value = float(value)
                if min_val <= numeric_value <= max_val:
                    return numeric_value
                elif default is not None:
                    return default
                else:
                    raise ValueError(f"Value {numeric_value} outside range [{min_val}, {max_val}]")
            except (TypeError, ValueError):
                if default is not None:
                    return default
                raise ValueError(f"Invalid numeric value: {value}")
        
        # Test valid ranges
        assert validate_range(0.5, 0.0, 1.0) == 0.5
        assert validate_range("0.7", 0.0, 1.0) == 0.7
        assert validate_range(10, 5, 15) == 10.0
        
        # Test boundary conditions
        assert validate_range(0.0, 0.0, 1.0) == 0.0  # Min boundary
        assert validate_range(1.0, 0.0, 1.0) == 1.0  # Max boundary
        
        # Test out of range with default
        assert validate_range(-0.5, 0.0, 1.0, default=0.0) == 0.0
        assert validate_range(1.5, 0.0, 1.0, default=1.0) == 1.0
        
        # Test invalid input with default
        assert validate_range("invalid", 0.0, 1.0, default=0.5) == 0.5
        assert validate_range(None, 0.0, 1.0, default=0.0) == 0.0
        
        print("✅ Range validation utilities passed")
    
    def test_type_checking_utilities(self):
        """Test type checking and conversion utilities"""
        
        def ensure_list(value: Any) -> List:
            """Ensure value is a list"""
            if isinstance(value, list):
                return value
            elif value is None:
                return []
            elif isinstance(value, (str, int, float, dict)):
                return [value]
            else:
                return list(value) if hasattr(value, '__iter__') else [value]
        
        # Test various inputs
        assert ensure_list([1, 2, 3]) == [1, 2, 3]
        assert ensure_list("string") == ["string"]
        assert ensure_list(42) == [42]
        assert ensure_list(None) == []
        assert ensure_list({"key": "value"}) == [{"key": "value"}]
        
        def ensure_dict(value: Any) -> Dict:
            """Ensure value is a dictionary"""
            if isinstance(value, dict):
                return value
            elif value is None:
                return {}
            else:
                return {"value": value}
        
        # Test dictionary conversion
        assert ensure_dict({"key": "value"}) == {"key": "value"}
        assert ensure_dict(None) == {}
        assert ensure_dict("string") == {"value": "string"}
        assert ensure_dict(42) == {"value": 42}
        
        print("✅ Type checking utilities validation passed")