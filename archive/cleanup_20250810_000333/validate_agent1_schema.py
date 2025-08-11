#!/usr/bin/env python3
"""
Agent 1 Schema Validation - Comprehensive Testing
================================================

This script runs Agent 1 (Domain Intelligence Agent) with real Azure services 
and real data to validate schema compliance.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Set up proper Python path
import sys
sys.path.insert(0, '/workspace/azure-maintie-rag')

from agents.domain_intelligence.agent import run_domain_analysis
from agents.core.universal_models import UniversalDomainAnalysis


class Agent1SchemaValidator:
    """Validates Agent 1 output against required schema."""
    
    def __init__(self):
        self.data_dir = Path("/workspace/azure-maintie-rag/data/raw/azure-ai-services-language-service_output")
        self.required_fields = self._get_required_schema()
        self.test_results = []
        
    def _get_required_schema(self) -> Dict[str, Any]:
        """Define the exact schema requirements for UniversalDomainAnalysis."""
        return {
            # Top-level fields
            "domain_signature": {"type": str, "required": True},
            "content_type_confidence": {"type": float, "required": True, "range": (0.0, 1.0)},
            "analysis_timestamp": {"type": str, "required": True},
            "processing_time": {"type": float, "required": True, "min": 0.0},
            "data_source_path": {"type": str, "required": True},
            "analysis_reliability": {"type": float, "required": True, "range": (0.0, 1.0)},
            "key_insights": {"type": list, "required": True, "item_type": str},
            "adaptation_recommendations": {"type": list, "required": True, "item_type": str},
            
            # characteristics fields
            "characteristics": {
                "type": "UniversalDomainCharacteristics", 
                "required": True,
                "fields": {
                    "avg_document_length": {"type": int, "required": True},
                    "document_count": {"type": int, "required": True},
                    "vocabulary_richness": {"type": float, "required": True, "range": (0.0, 1.0)},
                    "sentence_complexity": {"type": float, "required": True, "min": 0.0},
                    "most_frequent_terms": {"type": list, "required": True, "item_type": str},
                    "content_patterns": {"type": list, "required": True, "item_type": str},
                    "language_indicators": {"type": dict, "required": True},
                    "lexical_diversity": {"type": float, "required": True, "range": (0.0, 1.0)},
                    "vocabulary_complexity_ratio": {"type": float, "required": True, "range": (0.0, 1.0)},
                    "structural_consistency": {"type": float, "required": True, "range": (0.0, 1.0)},
                }
            },
            
            # processing_config fields
            "processing_config": {
                "type": "UniversalProcessingConfiguration",
                "required": True,
                "fields": {
                    "optimal_chunk_size": {"type": int, "required": True, "range": (100, 4000)},
                    "chunk_overlap_ratio": {"type": float, "required": True, "range": (0.0, 0.5)},
                    "entity_confidence_threshold": {"type": float, "required": True, "range": (0.5, 1.0)},
                    "relationship_density": {"type": float, "required": True, "range": (0.0, 1.0)},
                    "vector_search_weight": {"type": float, "required": True, "range": (0.0, 1.0)},
                    "graph_search_weight": {"type": float, "required": True, "range": (0.0, 1.0)},
                    "expected_extraction_quality": {"type": float, "required": True, "range": (0.0, 1.0)},
                    "processing_complexity": {"type": str, "required": True},
                }
            }
        }
    
    def validate_field(self, obj: Any, field_path: str, schema: Dict[str, Any]) -> List[str]:
        """Validate a single field against schema requirements."""
        errors = []
        
        if schema.get("required", False):
            if not hasattr(obj, field_path.split('.')[-1]):
                errors.append(f"Missing required field: {field_path}")
                return errors
        
        try:
            # Get the value using dot notation
            value = obj
            for part in field_path.split('.'):
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    errors.append(f"Field not accessible: {field_path}")
                    return errors
            
            # Type validation
            expected_type = schema["type"]
            if expected_type == str and not isinstance(value, str):
                errors.append(f"Field {field_path}: expected str, got {type(value)}")
            elif expected_type == float and not isinstance(value, (float, int)):
                errors.append(f"Field {field_path}: expected float, got {type(value)}")
            elif expected_type == int and not isinstance(value, int):
                errors.append(f"Field {field_path}: expected int, got {type(value)}")
            elif expected_type == list and not isinstance(value, list):
                errors.append(f"Field {field_path}: expected list, got {type(value)}")
            elif expected_type == dict and not isinstance(value, dict):
                errors.append(f"Field {field_path}: expected dict, got {type(value)}")
            
            # Range validation for numeric types
            if "range" in schema and isinstance(value, (int, float)):
                min_val, max_val = schema["range"]
                if not (min_val <= value <= max_val):
                    errors.append(f"Field {field_path}: value {value} not in range [{min_val}, {max_val}]")
            
            # Minimum value validation
            if "min" in schema and isinstance(value, (int, float)):
                if value < schema["min"]:
                    errors.append(f"Field {field_path}: value {value} less than minimum {schema['min']}")
            
            # List item type validation
            if "item_type" in schema and isinstance(value, list):
                item_type = schema["item_type"]
                for i, item in enumerate(value):
                    if item_type == str and not isinstance(item, str):
                        errors.append(f"Field {field_path}[{i}]: expected str, got {type(item)}")
            
        except Exception as e:
            errors.append(f"Field {field_path}: validation error - {str(e)}")
        
        return errors
    
    def validate_output(self, output: UniversalDomainAnalysis, file_path: str) -> Dict[str, Any]:
        """Validate complete Agent 1 output against schema."""
        validation_result = {
            "file_path": file_path,
            "validation_errors": [],
            "missing_fields": [],
            "incorrect_field_names": [],
            "field_validations": {},
            "compliance_score": 0.0
        }
        
        total_fields = 0
        valid_fields = 0
        
        # Validate top-level fields
        for field_name, schema in self.required_fields.items():
            if field_name in ["characteristics", "processing_config"]:
                continue  # Handle nested objects separately
                
            total_fields += 1
            errors = self.validate_field(output, field_name, schema)
            
            if errors:
                validation_result["validation_errors"].extend(errors)
                validation_result["field_validations"][field_name] = {"valid": False, "errors": errors}
            else:
                valid_fields += 1
                validation_result["field_validations"][field_name] = {"valid": True, "errors": []}
        
        # Validate characteristics nested object
        if hasattr(output, 'characteristics'):
            char_schema = self.required_fields["characteristics"]["fields"]
            for field_name, schema in char_schema.items():
                total_fields += 1
                field_path = f"characteristics.{field_name}"
                errors = self.validate_field(output, field_path, schema)
                
                if errors:
                    validation_result["validation_errors"].extend(errors)
                    validation_result["field_validations"][field_path] = {"valid": False, "errors": errors}
                else:
                    valid_fields += 1
                    validation_result["field_validations"][field_path] = {"valid": True, "errors": []}
        else:
            validation_result["missing_fields"].append("characteristics")
        
        # Validate processing_config nested object
        if hasattr(output, 'processing_config'):
            config_schema = self.required_fields["processing_config"]["fields"]
            for field_name, schema in config_schema.items():
                total_fields += 1
                field_path = f"processing_config.{field_name}"
                errors = self.validate_field(output, field_path, schema)
                
                if errors:
                    validation_result["validation_errors"].extend(errors)
                    validation_result["field_validations"][field_path] = {"valid": False, "errors": errors}
                else:
                    valid_fields += 1
                    validation_result["field_validations"][field_path] = {"valid": True, "errors": []}
        else:
            validation_result["missing_fields"].append("processing_config")
        
        # Check for critical field name issues
        if hasattr(output, 'characteristics'):
            # Check for incorrect vocabulary_complexity vs vocabulary_complexity_ratio
            if hasattr(output.characteristics, 'vocabulary_complexity'):
                if not hasattr(output.characteristics, 'vocabulary_complexity_ratio'):
                    validation_result["incorrect_field_names"].append(
                        "Found 'vocabulary_complexity' but schema requires 'vocabulary_complexity_ratio'"
                    )
        
        # Calculate compliance score
        validation_result["compliance_score"] = (valid_fields / total_fields * 100) if total_fields > 0 else 0
        
        return validation_result
    
    async def run_single_test(self, file_path: Path) -> Dict[str, Any]:
        """Run Agent 1 on a single file and validate output."""
        print(f"Testing file: {file_path.name}")
        
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            print(f"Content length: {len(content)} characters")
            
            # Run Agent 1
            start_time = time.time()
            result = await run_domain_analysis(content[:2000], detailed=True)  # Limit content for testing
            processing_time = time.time() - start_time
            
            print(f"Agent 1 completed in {processing_time:.2f} seconds")
            
            # Validate result
            validation = self.validate_output(result, str(file_path))
            validation["agent_processing_time"] = processing_time
            validation["agent_output_raw"] = result.model_dump() if hasattr(result, 'model_dump') else str(result)
            
            return validation
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            traceback.print_exc()
            return {
                "file_path": str(file_path),
                "error": str(e),
                "compliance_score": 0.0,
                "validation_errors": [f"Agent execution failed: {str(e)}"]
            }
    
    async def run_comprehensive_validation(self, max_files: int = 5) -> Dict[str, Any]:
        """Run comprehensive validation with multiple files."""
        print("=== Agent 1 Schema Validation - Comprehensive Testing ===\n")
        
        # Get test files
        test_files = list(self.data_dir.glob("*.md"))[:max_files]
        print(f"Testing with {len(test_files)} files from {self.data_dir}\n")
        
        results = {
            "total_files_tested": len(test_files),
            "test_results": [],
            "summary": {
                "files_passed": 0,
                "files_failed": 0,
                "average_compliance": 0.0,
                "common_errors": [],
                "missing_fields_summary": {},
                "field_validation_summary": {}
            }
        }
        
        # Run tests
        for file_path in test_files:
            test_result = await self.run_single_test(file_path)
            results["test_results"].append(test_result)
            
            # Update summary
            if test_result.get("compliance_score", 0) >= 95:
                results["summary"]["files_passed"] += 1
            else:
                results["summary"]["files_failed"] += 1
        
        # Calculate summary statistics
        compliance_scores = [r.get("compliance_score", 0) for r in results["test_results"]]
        results["summary"]["average_compliance"] = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0
        
        # Collect common errors
        all_errors = []
        for result in results["test_results"]:
            all_errors.extend(result.get("validation_errors", []))
        
        # Count error frequencies
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        results["summary"]["common_errors"] = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return results


async def main():
    """Main execution function."""
    validator = Agent1SchemaValidator()
    
    print("Starting Agent 1 Schema Validation...")
    print("This will test Agent 1 with real Azure OpenAI services and real data.\n")
    
    try:
        # Run comprehensive validation
        results = await validator.run_comprehensive_validation(max_files=3)
        
        # Print results
        print("\n=== VALIDATION RESULTS ===")
        print(f"Files tested: {results['total_files_tested']}")
        print(f"Files passed (>95% compliance): {results['summary']['files_passed']}")
        print(f"Files failed: {results['summary']['files_failed']}")
        print(f"Average compliance: {results['summary']['average_compliance']:.1f}%")
        
        print("\n=== DETAILED RESULTS ===")
        for i, test_result in enumerate(results['test_results'], 1):
            print(f"\nTest {i}: {Path(test_result['file_path']).name}")
            print(f"  Compliance Score: {test_result.get('compliance_score', 0):.1f}%")
            print(f"  Processing Time: {test_result.get('agent_processing_time', 0):.2f}s")
            
            if test_result.get('validation_errors'):
                print(f"  Validation Errors ({len(test_result['validation_errors'])}):")
                for error in test_result['validation_errors'][:5]:  # Show first 5 errors
                    print(f"    - {error}")
                if len(test_result['validation_errors']) > 5:
                    print(f"    ... and {len(test_result['validation_errors']) - 5} more errors")
            
            # Show a sample of the agent output
            if 'agent_output_raw' in test_result:
                output = test_result['agent_output_raw']
                print(f"  Sample Output Fields:")
                if isinstance(output, dict):
                    for key in ['domain_signature', 'content_type_confidence', 'analysis_timestamp'][:3]:
                        if key in output:
                            print(f"    {key}: {output[key]}")
        
        print("\n=== COMMON VALIDATION ERRORS ===")
        for error, count in results['summary']['common_errors'][:10]:
            print(f"  {error} (occurred {count} times)")
        
        # Save results to file
        output_file = Path("/workspace/azure-maintie-rag/agent1_schema_validation_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"Validation failed: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = asyncio.run(main())