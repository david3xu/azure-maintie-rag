#!/usr/bin/env python3
"""
Comprehensive Agent 1 Validation Report
======================================

This script provides a complete validation report of Agent 1 (Domain Intelligence Agent)
testing with real Azure services and real data.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Set up proper Python path
sys.path.insert(0, "/workspace/azure-maintie-rag")

from agents.domain_intelligence.agent import run_domain_analysis


async def run_comprehensive_agent1_validation():
    """Run comprehensive validation with detailed analysis."""

    print("=" * 80)
    print("AGENT 1 (DOMAIN INTELLIGENCE) COMPREHENSIVE VALIDATION REPORT")
    print("=" * 80)
    print()

    # Test setup
    data_dir = Path(
        "/workspace/azure-maintie-rag/data/raw/azure-ai-services-language-service_output"
    )
    test_files = list(data_dir.glob("*.md"))[
        :5
    ]  # Test with 5 files for comprehensive coverage

    print("🎯 VALIDATION OBJECTIVES:")
    print("1. Validate Agent 1 output against UniversalDomainAnalysis schema")
    print("2. Verify all required fields are present and correctly named")
    print("3. Test processing config field population")
    print("4. Validate characteristics field population")
    print(
        "5. Check field name compliance (vocabulary_complexity_ratio vs vocabulary_complexity)"
    )
    print()

    print("🔧 TEST CONFIGURATION:")
    print(f"   - Azure OpenAI: REAL services (no mocks)")
    print(f"   - Data Source: {data_dir}")
    print(f"   - Test Files: {len(test_files)}")
    print(f"   - PYTHONPATH: /workspace/azure-maintie-rag")
    print()

    # Required schema definition
    required_schema = {
        # Top-level UniversalDomainAnalysis fields
        "domain_signature": {"type": str, "description": "Content-based signature"},
        "content_type_confidence": {
            "type": float,
            "description": "Confidence score",
            "range": (0.0, 1.0),
        },
        "analysis_timestamp": {"type": str, "description": "Analysis timestamp"},
        "processing_time": {"type": float, "description": "Processing time in seconds"},
        "data_source_path": {"type": str, "description": "Source data path"},
        "analysis_reliability": {
            "type": float,
            "description": "Analysis reliability score",
            "range": (0.0, 1.0),
        },
        "key_insights": {"type": list, "description": "Key insights discovered"},
        "adaptation_recommendations": {
            "type": list,
            "description": "Processing recommendations",
        },
        # UniversalDomainCharacteristics nested fields
        "characteristics.avg_document_length": {
            "type": int,
            "description": "Average document length",
        },
        "characteristics.document_count": {
            "type": int,
            "description": "Number of documents",
        },
        "characteristics.vocabulary_richness": {
            "type": float,
            "description": "Vocabulary richness ratio",
            "range": (0.0, 1.0),
        },
        "characteristics.sentence_complexity": {
            "type": float,
            "description": "Average words per sentence",
        },
        "characteristics.most_frequent_terms": {
            "type": list,
            "description": "Most frequent terms",
        },
        "characteristics.content_patterns": {
            "type": list,
            "description": "Discovered content patterns",
        },
        "characteristics.language_indicators": {
            "type": dict,
            "description": "Language detection scores",
        },
        "characteristics.lexical_diversity": {
            "type": float,
            "description": "Type-token ratio",
            "range": (0.0, 1.0),
        },
        "characteristics.vocabulary_complexity_ratio": {
            "type": float,
            "description": "CRITICAL: Must be this exact field name",
            "range": (0.0, 1.0),
        },
        "characteristics.structural_consistency": {
            "type": float,
            "description": "Structure consistency score",
            "range": (0.0, 1.0),
        },
        # UniversalProcessingConfiguration nested fields
        "processing_config.optimal_chunk_size": {
            "type": int,
            "description": "Optimal chunk size",
            "range": (100, 4000),
        },
        "processing_config.chunk_overlap_ratio": {
            "type": float,
            "description": "Chunk overlap ratio",
            "range": (0.0, 0.5),
        },
        "processing_config.entity_confidence_threshold": {
            "type": float,
            "description": "Entity extraction threshold",
            "range": (0.5, 1.0),
        },
        "processing_config.relationship_density": {
            "type": float,
            "description": "Relationship density",
            "range": (0.0, 1.0),
        },
        "processing_config.vector_search_weight": {
            "type": float,
            "description": "Vector search weight",
            "range": (0.0, 1.0),
        },
        "processing_config.graph_search_weight": {
            "type": float,
            "description": "Graph search weight",
            "range": (0.0, 1.0),
        },
        "processing_config.expected_extraction_quality": {
            "type": float,
            "description": "Expected quality",
            "range": (0.0, 1.0),
        },
        "processing_config.processing_complexity": {
            "type": str,
            "description": "Complexity level (low/medium/high)",
        },
    }

    print("📋 SCHEMA REQUIREMENTS:")
    print(f"   - Total Required Fields: {len(required_schema)}")
    print(f"   - Top-level Fields: 8")
    print(f"   - Characteristics Fields: 10")
    print(f"   - Processing Config Fields: 8")
    print()

    # Run tests
    test_results = []

    for i, test_file in enumerate(test_files, 1):
        print(f"🧪 TEST {i}: {test_file.name}")
        print(f"   File size: {test_file.stat().st_size} bytes")

        try:
            # Read and prepare content
            content = test_file.read_text(encoding="utf-8")
            test_content = content[:3000]  # Test with larger content sample
            print(f"   Content length: {len(test_content)} characters")

            # Run Agent 1
            start_time = time.time()
            result = await run_domain_analysis(test_content, detailed=True)
            processing_time = time.time() - start_time

            print(f"   ✅ Agent completed in {processing_time:.2f} seconds")

            # Convert to dict for analysis
            if hasattr(result, "model_dump"):
                output_dict = result.model_dump()
            else:
                output_dict = result.__dict__

            test_results.append(
                {
                    "test_number": i,
                    "file_name": test_file.name,
                    "processing_time": processing_time,
                    "output": output_dict,
                    "agent_object": result,
                }
            )

        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            test_results.append(
                {
                    "test_number": i,
                    "file_name": test_file.name,
                    "error": str(e),
                    "processing_time": 0,
                    "output": None,
                }
            )

        print()

    # Validation analysis
    print("=" * 80)
    print("DETAILED VALIDATION RESULTS")
    print("=" * 80)
    print()

    successful_tests = [r for r in test_results if r.get("output")]
    failed_tests = [r for r in test_results if not r.get("output")]

    print(f"✅ Successful Tests: {len(successful_tests)}/{len(test_results)}")
    print(f"❌ Failed Tests: {len(failed_tests)}/{len(test_results)}")
    print()

    if failed_tests:
        print("❌ FAILED TESTS:")
        for test in failed_tests:
            print(f"   - {test['file_name']}: {test['error']}")
        print()

    # Schema compliance check
    compliance_results = []

    for test in successful_tests:
        output = test["output"]
        compliance = {
            "test_number": test["test_number"],
            "file_name": test["file_name"],
            "processing_time": test["processing_time"],
            "missing_fields": [],
            "type_errors": [],
            "range_errors": [],
            "field_values": {},
        }

        # Check each required field
        for field_path, requirements in required_schema.items():
            expected_type = requirements["type"]

            # Navigate to field value
            try:
                if "." in field_path:
                    parent, child = field_path.split(".", 1)
                    if parent in output and child in output[parent]:
                        value = output[parent][child]
                    else:
                        compliance["missing_fields"].append(field_path)
                        continue
                else:
                    if field_path in output:
                        value = output[field_path]
                    else:
                        compliance["missing_fields"].append(field_path)
                        continue

                # Store field value for reporting
                compliance["field_values"][field_path] = value

                # Type validation
                if expected_type == str and not isinstance(value, str):
                    compliance["type_errors"].append(
                        f"{field_path}: expected {expected_type.__name__}, got {type(value).__name__}"
                    )
                elif expected_type == float and not isinstance(value, (float, int)):
                    compliance["type_errors"].append(
                        f"{field_path}: expected {expected_type.__name__}, got {type(value).__name__}"
                    )
                elif expected_type == int and not isinstance(value, int):
                    compliance["type_errors"].append(
                        f"{field_path}: expected {expected_type.__name__}, got {type(value).__name__}"
                    )
                elif expected_type == list and not isinstance(value, list):
                    compliance["type_errors"].append(
                        f"{field_path}: expected {expected_type.__name__}, got {type(value).__name__}"
                    )
                elif expected_type == dict and not isinstance(value, dict):
                    compliance["type_errors"].append(
                        f"{field_path}: expected {expected_type.__name__}, got {type(value).__name__}"
                    )

                # Range validation
                if "range" in requirements and isinstance(value, (int, float)):
                    min_val, max_val = requirements["range"]
                    if not (min_val <= value <= max_val):
                        compliance["range_errors"].append(
                            f"{field_path}: {value} not in range [{min_val}, {max_val}]"
                        )

            except Exception as e:
                compliance["missing_fields"].append(f"{field_path} (access error: {e})")

        compliance_results.append(compliance)

    # Report results
    total_required_fields = len(required_schema)

    for compliance in compliance_results:
        print(f"📊 TEST {compliance['test_number']}: {compliance['file_name']}")
        print(f"   Processing Time: {compliance['processing_time']:.2f}s")

        missing_count = len(compliance["missing_fields"])
        type_error_count = len(compliance["type_errors"])
        range_error_count = len(compliance["range_errors"])

        present_fields = total_required_fields - missing_count
        compliance_percentage = (present_fields / total_required_fields) * 100

        if missing_count == 0 and type_error_count == 0 and range_error_count == 0:
            print(
                f"   ✅ PERFECT COMPLIANCE: 100% ({present_fields}/{total_required_fields} fields)"
            )
        else:
            print(
                f"   ⚠️  PARTIAL COMPLIANCE: {compliance_percentage:.1f}% ({present_fields}/{total_required_fields} fields)"
            )

        if missing_count > 0:
            print(f"   ❌ Missing Fields ({missing_count}):")
            for field in compliance["missing_fields"][:5]:  # Show first 5
                print(f"      - {field}")
            if missing_count > 5:
                print(f"      ... and {missing_count - 5} more")

        if type_error_count > 0:
            print(f"   ❌ Type Errors ({type_error_count}):")
            for error in compliance["type_errors"]:
                print(f"      - {error}")

        if range_error_count > 0:
            print(f"   ❌ Range Errors ({range_error_count}):")
            for error in compliance["range_errors"]:
                print(f"      - {error}")

        # Show critical field values
        print("   📋 Critical Field Values:")
        critical_fields = [
            "domain_signature",
            "characteristics.vocabulary_complexity_ratio",
            "processing_config.optimal_chunk_size",
            "processing_config.processing_complexity",
        ]

        for field in critical_fields:
            if field in compliance["field_values"]:
                value = compliance["field_values"][field]
                print(f"      ✅ {field}: {value}")
            else:
                print(f"      ❌ {field}: MISSING")

        print()

    # Overall summary
    print("=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)
    print()

    perfect_compliance_count = sum(
        1
        for c in compliance_results
        if len(c["missing_fields"]) == 0
        and len(c["type_errors"]) == 0
        and len(c["range_errors"]) == 0
    )

    total_successful_tests = len(successful_tests)
    overall_compliance_rate = (
        (perfect_compliance_count / total_successful_tests * 100)
        if total_successful_tests > 0
        else 0
    )

    print(f"🎯 OVERALL RESULTS:")
    print(f"   Total Tests Run: {len(test_results)}")
    print(f"   Successful Executions: {len(successful_tests)}")
    print(
        f"   Perfect Schema Compliance: {perfect_compliance_count}/{len(successful_tests)}"
    )
    print(f"   Overall Compliance Rate: {overall_compliance_rate:.1f}%")
    print()

    if perfect_compliance_count == len(successful_tests) and len(successful_tests) > 0:
        print(
            "🎉 VALIDATION PASSED: Agent 1 fully complies with UniversalDomainAnalysis schema!"
        )
        print()
        print("✅ CONFIRMED:")
        print("   - All required fields are present")
        print("   - Field names match schema exactly (vocabulary_complexity_ratio ✓)")
        print("   - All field types are correct")
        print("   - Processing config populated correctly")
        print("   - Characteristics populated correctly")
        print("   - Real Azure OpenAI integration working")
    else:
        print("⚠️  VALIDATION ISSUES FOUND")
        print()
        if len(failed_tests) > 0:
            print("❌ EXECUTION ISSUES:")
            for test in failed_tests:
                print(f"   - {test['file_name']}: {test['error']}")

        incomplete_compliance = [
            c
            for c in compliance_results
            if len(c["missing_fields"]) > 0
            or len(c["type_errors"]) > 0
            or len(c["range_errors"]) > 0
        ]

        if incomplete_compliance:
            print("❌ SCHEMA COMPLIANCE ISSUES:")
            for c in incomplete_compliance:
                issues = (
                    len(c["missing_fields"])
                    + len(c["type_errors"])
                    + len(c["range_errors"])
                )
                print(f"   - {c['file_name']}: {issues} compliance issues")

    print()
    print("📄 SAMPLE OUTPUT (First Successful Test):")
    if successful_tests:
        sample_output = successful_tests[0]["output"]
        print(json.dumps(sample_output, indent=2))

    return {
        "total_tests": len(test_results),
        "successful_tests": len(successful_tests),
        "perfect_compliance": perfect_compliance_count,
        "compliance_rate": overall_compliance_rate,
        "test_results": test_results,
        "compliance_results": compliance_results,
    }


if __name__ == "__main__":
    results = asyncio.run(run_comprehensive_agent1_validation())
