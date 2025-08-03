#!/usr/bin/env python3
"""
Test Config-Extraction Orchestration Workflow

Tests the complete two-stage architecture implementation:
1. Domain Intelligence Agent generates ExtractionConfiguration
2. Knowledge Extraction Agent processes documents using configuration

This validates the CONFIG_VS_EXTRACTION_ARCHITECTURE implementation.
"""

import asyncio
import tempfile
from pathlib import Path
from agents.config_extraction_orchestrator import ConfigExtractionOrchestrator, process_domain_with_config_extraction

async def test_config_extraction_workflow():
    """Test the complete Config-Extraction workflow"""
    print("🚀 Testing Config-Extraction Orchestration Workflow")
    print("=" * 60)
    
    # Create temporary test domain directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        domain_dir = temp_path / "test_programming"
        domain_dir.mkdir()
        
        # Create test documents
        doc1 = domain_dir / "functions.md"
        doc1.write_text("""
# Programming Functions

Functions are reusable blocks of code that perform specific tasks.

## Function Definition
def calculate_sum(a, b):
    return a + b

## Classes and Objects
class Calculator:
    def add(self, x, y):
        return x + y
    
    def multiply(self, x, y):
        return x * y

## Variables and Data Types
- Integer: 42
- String: "hello world"
- List: [1, 2, 3, 4, 5]
- Dictionary: {"name": "John", "age": 30}

## Control Flow
if condition:
    execute_code()
elif other_condition:
    execute_other_code()
else:
    execute_default()

for item in items:
    process(item)

while running:
    continue_execution()
""")
        
        doc2 = domain_dir / "algorithms.md"
        doc2.write_text("""
# Programming Algorithms

Algorithms are step-by-step procedures for solving problems.

## Sorting Algorithms
- Bubble Sort: Simple comparison-based algorithm
- Quick Sort: Divide-and-conquer approach
- Merge Sort: Stable sorting algorithm

## Search Algorithms
- Linear Search: Sequential checking
- Binary Search: Divide-and-conquer on sorted data
- Hash Search: Direct access via hash function

## Data Structures
- Array: Contiguous memory storage
- Linked List: Dynamic node-based structure
- Tree: Hierarchical data organization
- Graph: Network of connected nodes

## Algorithm Analysis
- Time Complexity: O(n), O(log n), O(n²)
- Space Complexity: Memory usage analysis
- Big O Notation: Asymptotic behavior description
""")
        
        print(f"📁 Created test domain directory: {domain_dir}")
        print(f"📄 Created {len(list(domain_dir.glob('*.md')))} test documents")
        print()
        
        try:
            # Test the complete workflow
            print("🔄 Stage 1: Testing Domain Intelligence Agent...")
            orchestrator = ConfigExtractionOrchestrator()
            
            # Test configuration generation
            extraction_config = await orchestrator._generate_extraction_configuration(
                "test_programming", domain_dir, force_regenerate=True
            )
            
            if extraction_config:
                print("✅ Stage 1 Complete: ExtractionConfiguration generated")
                print(f"   Domain: {extraction_config.domain_name}")
                print(f"   Entity threshold: {extraction_config.entity_confidence_threshold}")
                print(f"   Expected entity types: {len(extraction_config.expected_entity_types)}")
                print(f"   Relationship patterns: {len(extraction_config.relationship_patterns)}")
                print(f"   Processing strategy: {extraction_config.processing_strategy}")
                print(f"   Chunk size: {extraction_config.chunk_size}")
                print()
            else:
                print("❌ Stage 1 Failed: Could not generate ExtractionConfiguration")
                return False
            
            print("🔄 Stage 2: Testing Knowledge Extraction Agent...")
            
            # Test knowledge extraction
            extraction_results = await orchestrator._extract_knowledge_with_config(
                domain_dir, extraction_config
            )
            
            if extraction_results:
                print("✅ Stage 2 Complete: Knowledge extraction completed")
                print(f"   Documents processed: {extraction_results.documents_processed}")
                print(f"   Extraction accuracy: {extraction_results.extraction_accuracy:.2f}")
                print(f"   Entities extracted: {extraction_results.total_entities_extracted}")
                print(f"   Relationships extracted: {extraction_results.total_relationships_extracted}")
                print(f"   Validation passed: {extraction_results.extraction_passed_validation}")
                print()
            else:
                print("❌ Stage 2 Failed: Knowledge extraction failed")
                return False
            
            print("🔄 Testing Complete Workflow...")
            
            # Test the complete process using convenience function
            complete_results = await process_domain_with_config_extraction(
                domain_dir, force_regenerate_config=True
            )
            
            if complete_results and complete_results.get("workflow_status") == "completed":
                print("🎉 Complete Workflow Test PASSED")
                print(f"   Domain: {complete_results['domain_name']}")
                print(f"   Stage 1 complete: {complete_results['stage_1_complete']}")
                print(f"   Stage 2 complete: {complete_results['stage_2_complete']}")
                print(f"   Total entities: {complete_results['extraction_results'].total_entities_extracted}")
                print(f"   Total relationships: {complete_results['extraction_results'].total_relationships_extracted}")
                return True
            else:
                print("❌ Complete Workflow Test FAILED")
                return False
                
        except Exception as e:
            print(f"❌ Test Error: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_error_handling():
    """Test error handling in the orchestration workflow"""
    print("\n🔧 Testing Error Handling")
    print("=" * 40)
    
    orchestrator = ConfigExtractionOrchestrator()
    
    # Test with non-existent domain
    try:
        result = await orchestrator.process_domain_documents(
            Path("/non/existent/path"), force_regenerate_config=True
        )
        print("❌ Should have failed for non-existent path")
        return False
    except Exception as e:
        print(f"✅ Correctly handled non-existent path: {type(e).__name__}")
    
    # Test with empty domain directory
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_dir = Path(temp_dir) / "empty_domain"
        empty_dir.mkdir()
        
        try:
            extraction_results = await orchestrator._extract_knowledge_with_config(
                empty_dir, 
                # Create minimal config for testing
                type('MockConfig', (), {
                    'domain_name': 'empty_test',
                    'entity_confidence_threshold': 0.7,
                    'expected_entity_types': [],
                    'relationship_patterns': [],
                    'processing_strategy': 'technical_content',
                    'chunk_size': 1000,
                    'chunk_overlap': 200,
                    'technical_vocabulary': [],
                    'key_concepts': [],
                    'minimum_quality_score': 0.6,
                    'validation_criteria': {},
                    'enable_caching': False,
                    'cache_ttl_seconds': 3600,
                    'max_entities_per_chunk': 50,
                    'max_relationships_per_chunk': 30
                })()
            )
            
            if (extraction_results and 
                extraction_results.documents_processed == 0 and 
                extraction_results.total_entities_extracted == 0):
                print("✅ Correctly handled empty domain directory")
                return True
            else:
                print("❌ Empty domain handling failed")
                return False
                
        except Exception as e:
            print(f"⚠️ Unexpected error with empty domain: {e}")
            return False

async def main():
    """Run all tests"""
    print("Config-Extraction Orchestration Integration Test")
    print("=" * 60)
    
    # Test main workflow
    workflow_success = await test_config_extraction_workflow()
    
    # Test error handling
    error_handling_success = await test_error_handling()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Main Workflow: {'✅ PASSED' if workflow_success else '❌ FAILED'}")
    print(f"Error Handling: {'✅ PASSED' if error_handling_success else '❌ FAILED'}")
    
    overall_success = workflow_success and error_handling_success
    print(f"\nOverall Result: {'🎉 ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n✅ Config-Extraction Architecture Implementation VALIDATED")
        print("   The two-stage workflow is working correctly:")
        print("   1. Domain Intelligence Agent → ExtractionConfiguration")
        print("   2. Knowledge Extraction Agent → ExtractionResults")
        print("   Architecture from CONFIG_VS_EXTRACTION_ARCHITECTURE.md is implemented!")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)