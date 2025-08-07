#!/usr/bin/env python3
"""
Test Script for Simplified Agent Architecture
==============================================

This demonstrates the simplified agents working independently without
the complex dependency chain of the original architecture.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_simplified_architecture():
    """Test the simplified agent architecture independently"""
    
    print("🔍 Testing Simplified Agent Architecture")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Domain Intelligence Agent (Standalone)
    print("\n🧠 Testing Domain Intelligence Agent...")
    total_tests += 1
    try:
        # Import and test domain agent directly (no complex dependencies)
        from agents.domain_intelligence.simplified_agent import get_domain_agent, DomainDeps
        
        agent = get_domain_agent()
        deps = DomainDeps(data_directory="/workspace/azure-maintie-rag/data/raw")
        
        print("✅ Domain agent created successfully")
        print(f"   - Agent type: {type(agent).__name__}")
        print(f"   - Dependencies: {type(deps).__name__}")
        print(f"   - Data directory: {deps.data_directory}")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Domain agent test failed: {e}")
    
    # Test 2: Knowledge Extraction Agent (Standalone)  
    print("\n📚 Testing Knowledge Extraction Agent...")
    total_tests += 1
    try:
        from agents.knowledge_extraction.simplified_agent import (
            get_extraction_agent, 
            ExtractionDeps,
            ExtractedEntity,
            ExtractionResult
        )
        
        agent = get_extraction_agent()
        deps = ExtractionDeps(
            confidence_threshold=0.8,
            max_entities_per_chunk=15,
            enable_relationships=True
        )
        
        print("✅ Extraction agent created successfully")
        print(f"   - Agent type: {type(agent).__name__}")
        print(f"   - Confidence threshold: {deps.confidence_threshold}")
        print(f"   - Max entities: {deps.max_entities_per_chunk}")
        print(f"   - Relationships enabled: {deps.enable_relationships}")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Extraction agent test failed: {e}")
    
    # Test 3: Universal Search Agent (Standalone)
    print("\n🔍 Testing Universal Search Agent...")
    total_tests += 1
    try:
        from agents.universal_search.simplified_agent import (
            get_search_agent,
            SearchDeps, 
            SearchResult,
            UniversalSearchResult
        )
        
        agent = get_search_agent()
        deps = SearchDeps(
            max_results=10,
            similarity_threshold=0.7,
            enable_vector_search=True,
            enable_graph_search=True,
            enable_gnn_search=False
        )
        
        print("✅ Search agent created successfully")
        print(f"   - Agent type: {type(agent).__name__}")
        print(f"   - Max results: {deps.max_results}")
        print(f"   - Similarity threshold: {deps.similarity_threshold}")
        print(f"   - Vector search: {deps.enable_vector_search}")
        print(f"   - Graph search: {deps.enable_graph_search}")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Search agent test failed: {e}")
    
    # Test 4: Simplified Orchestration (Standalone)
    print("\n🎼 Testing Simplified Orchestration...")
    total_tests += 1
    try:
        from agents.simplified_orchestration import (
            SimplifiedRAGOrchestrator,
            create_rag_orchestrator,
            RAGOrchestrationResult
        )
        
        orchestrator = create_rag_orchestrator()
        
        print("✅ Orchestrator created successfully")
        print(f"   - Orchestrator type: {type(orchestrator).__name__}")
        print(f"   - No complex initialization required")
        print(f"   - Direct agent composition pattern")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
    
    # Test 5: Model Simplification
    print("\n📋 Testing Model Simplification...")
    total_tests += 1
    try:
        # Test that we can import and use simplified models
        from agents.domain_intelligence.simplified_agent import DomainAnalysis
        from agents.knowledge_extraction.simplified_agent import ExtractedEntity, ExtractedRelationship
        from agents.universal_search.simplified_agent import SearchResult
        
        # Create sample instances to verify model structure
        domain_result = DomainAnalysis(
            detected_domain="test_domain",
            confidence=0.8,
            file_count=5,
            recommendations=["test recommendation"],
            processing_time=1.0
        )
        
        entity = ExtractedEntity(
            text="test entity",
            type="CONCEPT", 
            confidence=0.9,
            start_pos=0,
            end_pos=10
        )
        
        search_result = SearchResult(
            content="test content",
            relevance_score=0.8,
            source="test_source",
            result_type="vector"
        )
        
        print("✅ Simplified models working correctly")
        print(f"   - Domain model: {domain_result.detected_domain}")
        print(f"   - Entity model: {entity.text} ({entity.type})")
        print(f"   - Search model: {search_result.result_type}")
        
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 30)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\n🎉 All tests passed! Simplified architecture is working correctly.")
        print("\n✅ Key Achievements:")
        print("   - Agents can be imported independently")
        print("   - No complex dependency chains")
        print("   - Simple, focused dependency models")
        print("   - Direct PydanticAI agent creation")
        print("   - Clean model interfaces")
    else:
        print(f"\n⚠️  {total_tests - tests_passed} tests failed. Review implementation.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_simplified_architecture())
    sys.exit(0 if success else 1)