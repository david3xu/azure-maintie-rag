#!/usr/bin/env python3
"""
Fixed Agent 1 Schema Validation - Proper Field Access
====================================================

This script properly validates Agent 1 output against schema requirements.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def analyze_agent1_output():
    """Analyze the actual Agent 1 output to validate schema compliance."""

    # Load the validation results
    results_file = Path(
        "/workspace/azure-maintie-rag/agent1_schema_validation_results.json"
    )
    with open(results_file, "r") as f:
        results = json.load(f)

    print("=== AGENT 1 SCHEMA COMPLIANCE ANALYSIS ===\n")

    # Define required schema
    required_schema = {
        # Top-level fields
        "domain_signature": str,
        "content_type_confidence": float,
        "analysis_timestamp": str,
        "processing_time": float,
        "data_source_path": str,
        "analysis_reliability": float,
        "key_insights": list,
        "adaptation_recommendations": list,
        # Characteristics fields (nested)
        "characteristics": {
            "avg_document_length": int,
            "document_count": int,
            "vocabulary_richness": float,
            "sentence_complexity": float,
            "most_frequent_terms": list,
            "content_patterns": list,
            "language_indicators": dict,
            "lexical_diversity": float,
            "vocabulary_complexity_ratio": float,  # CRITICAL: This is the correct field name
            "structural_consistency": float,
        },
        # Processing config fields (nested)
        "processing_config": {
            "optimal_chunk_size": int,
            "chunk_overlap_ratio": float,
            "entity_confidence_threshold": float,
            "relationship_density": float,
            "vector_search_weight": float,
            "graph_search_weight": float,
            "expected_extraction_quality": float,
            "processing_complexity": str,
        },
    }

    def validate_single_output(output_data: Dict, test_name: str) -> Dict[str, Any]:
        """Validate a single output against schema requirements."""
        validation_result = {
            "test_name": test_name,
            "all_required_fields_present": True,
            "missing_fields": [],
            "field_type_errors": [],
            "field_name_issues": [],
            "compliance_details": {},
        }

        def check_field(
            data: Dict, field_path: str, expected_type: Any, parent_path: str = ""
        ):
            """Recursively check fields in nested structure."""
            full_path = f"{parent_path}.{field_path}" if parent_path else field_path

            if field_path not in data:
                validation_result["missing_fields"].append(full_path)
                validation_result["all_required_fields_present"] = False
                return False

            value = data[field_path]

            # Type checking
            if expected_type == str and not isinstance(value, str):
                validation_result["field_type_errors"].append(
                    f"{full_path}: expected str, got {type(value).__name__}"
                )
            elif expected_type == float and not isinstance(value, (float, int)):
                validation_result["field_type_errors"].append(
                    f"{full_path}: expected float, got {type(value).__name__}"
                )
            elif expected_type == int and not isinstance(value, int):
                validation_result["field_type_errors"].append(
                    f"{full_path}: expected int, got {type(value).__name__}"
                )
            elif expected_type == list and not isinstance(value, list):
                validation_result["field_type_errors"].append(
                    f"{full_path}: expected list, got {type(value).__name__}"
                )
            elif expected_type == dict and not isinstance(value, dict):
                validation_result["field_type_errors"].append(
                    f"{full_path}: expected dict, got {type(value).__name__}"
                )

            validation_result["compliance_details"][full_path] = {
                "present": True,
                "type": type(value).__name__,
                "expected_type": (
                    expected_type.__name__
                    if hasattr(expected_type, "__name__")
                    else str(expected_type)
                ),
                "value_sample": (
                    str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                ),
            }
            return True

        # Check top-level fields
        for field_name, expected_type in required_schema.items():
            if field_name in ["characteristics", "processing_config"]:
                continue  # Handle nested objects separately
            check_field(output_data, field_name, expected_type)

        # Check characteristics nested fields
        if "characteristics" in output_data:
            characteristics = output_data["characteristics"]
            for field_name, expected_type in required_schema["characteristics"].items():
                check_field(
                    characteristics, field_name, expected_type, "characteristics"
                )
        else:
            validation_result["missing_fields"].append("characteristics")
            validation_result["all_required_fields_present"] = False

        # Check processing_config nested fields
        if "processing_config" in output_data:
            processing_config = output_data["processing_config"]
            for field_name, expected_type in required_schema[
                "processing_config"
            ].items():
                check_field(
                    processing_config, field_name, expected_type, "processing_config"
                )
        else:
            validation_result["missing_fields"].append("processing_config")
            validation_result["all_required_fields_present"] = False

        # Check for critical field name issue
        if "characteristics" in output_data:
            chars = output_data["characteristics"]
            if (
                "vocabulary_complexity" in chars
                and "vocabulary_complexity_ratio" not in chars
            ):
                validation_result["field_name_issues"].append(
                    "CRITICAL: Found 'vocabulary_complexity' but schema requires 'vocabulary_complexity_ratio'"
                )
            elif "vocabulary_complexity_ratio" in chars:
                validation_result["compliance_details"][
                    "characteristics.vocabulary_complexity_ratio"
                ]["note"] = "CORRECT field name used"

        return validation_result

    # Validate each test result
    total_tests = len(results["test_results"])
    perfect_compliance_count = 0

    print(f"Validating {total_tests} test results...\n")

    for i, test_result in enumerate(results["test_results"], 1):
        if "agent_output_raw" not in test_result:
            print(f"Test {i}: No agent output found (execution failed)")
            continue

        output_data = test_result["agent_output_raw"]
        test_name = Path(test_result["file_path"]).name

        validation = validate_single_output(output_data, test_name)

        print(f"=== TEST {i}: {validation['test_name']} ===")
        print(
            f"Processing Time: {test_result.get('agent_processing_time', 'Unknown'):.2f}s"
        )

        if (
            validation["all_required_fields_present"]
            and not validation["field_type_errors"]
            and not validation["field_name_issues"]
        ):
            print(
                "✅ PERFECT COMPLIANCE - All required fields present with correct types"
            )
            perfect_compliance_count += 1
        else:
            print("❌ COMPLIANCE ISSUES FOUND")

        # Show missing fields
        if validation["missing_fields"]:
            print(f"\n❌ Missing Fields ({len(validation['missing_fields'])}):")
            for field in validation["missing_fields"]:
                print(f"   - {field}")
        else:
            print("\n✅ All Required Fields Present")

        # Show type errors
        if validation["field_type_errors"]:
            print(f"\n❌ Type Errors ({len(validation['field_type_errors'])}):")
            for error in validation["field_type_errors"]:
                print(f"   - {error}")
        else:
            print("\n✅ All Field Types Correct")

        # Show field name issues
        if validation["field_name_issues"]:
            print(f"\n❌ Field Name Issues ({len(validation['field_name_issues'])}):")
            for issue in validation["field_name_issues"]:
                print(f"   - {issue}")
        else:
            print("\n✅ All Field Names Correct")

        # Show critical fields validation
        print("\n📋 Critical Fields Validation:")
        critical_fields = [
            "domain_signature",
            "characteristics.vocabulary_complexity_ratio",
            "processing_config.optimal_chunk_size",
            "processing_config.processing_complexity",
        ]

        for field in critical_fields:
            if field in validation["compliance_details"]:
                details = validation["compliance_details"][field]
                print(f"   ✅ {field}: {details['type']} = {details['value_sample']}")
            else:
                print(f"   ❌ {field}: MISSING")

        print("\n" + "=" * 60)

    # Overall summary
    print(f"\n=== OVERALL VALIDATION SUMMARY ===")
    print(f"Total Tests: {total_tests}")
    print(f"Perfect Compliance: {perfect_compliance_count}/{total_tests}")
    print(f"Compliance Rate: {(perfect_compliance_count/total_tests)*100:.1f}%")

    if perfect_compliance_count == total_tests:
        print(
            "\n🎉 ALL TESTS PASSED - Agent 1 fully complies with schema requirements!"
        )
    else:
        print(
            f"\n⚠️  {total_tests - perfect_compliance_count} tests have compliance issues"
        )

    # Detailed field analysis
    print("\n=== DETAILED FIELD ANALYSIS ===")

    # Analyze all outputs to see field consistency
    all_fields_found = set()
    for test_result in results["test_results"]:
        if "agent_output_raw" in test_result:
            output_data = test_result["agent_output_raw"]

            # Top-level fields
            all_fields_found.update(output_data.keys())

            # Characteristics fields
            if "characteristics" in output_data:
                for field in output_data["characteristics"].keys():
                    all_fields_found.add(f"characteristics.{field}")

            # Processing config fields
            if "processing_config" in output_data:
                for field in output_data["processing_config"].keys():
                    all_fields_found.add(f"processing_config.{field}")

    # Required vs Found fields comparison
    all_required_fields = set()

    # Add top-level required fields
    for field in required_schema.keys():
        if field not in ["characteristics", "processing_config"]:
            all_required_fields.add(field)

    # Add nested required fields
    for field in required_schema["characteristics"].keys():
        all_required_fields.add(f"characteristics.{field}")

    for field in required_schema["processing_config"].keys():
        all_required_fields.add(f"processing_config.{field}")

    missing_from_output = all_required_fields - all_fields_found
    extra_in_output = all_fields_found - all_required_fields

    print(f"\nRequired Fields: {len(all_required_fields)}")
    print(f"Fields Found in Output: {len(all_fields_found)}")
    print(f"Missing from Output: {len(missing_from_output)}")
    print(f"Extra in Output: {len(extra_in_output)}")

    if missing_from_output:
        print(f"\n❌ Missing Required Fields:")
        for field in sorted(missing_from_output):
            print(f"   - {field}")

    if extra_in_output:
        print(f"\n➕ Additional Fields (not required but present):")
        for field in sorted(extra_in_output):
            print(f"   - {field}")

    return {
        "total_tests": total_tests,
        "perfect_compliance_count": perfect_compliance_count,
        "compliance_rate": (perfect_compliance_count / total_tests) * 100,
        "missing_required_fields": list(missing_from_output),
        "extra_fields": list(extra_in_output),
    }


if __name__ == "__main__":
    results = analyze_agent1_output()
